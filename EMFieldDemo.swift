// =============================================================================
//  EMFieldDemo.swift  —  GPU particle-swarm edition
//
//  A swarm of hundreds of thousands of charged test particles pushed through
//  electromagnetic fields on the GPU (Boris pusher in a compute kernel),
//  rendered as additive HDR point sprites coloured by speed, with a
//  feedback-accumulation buffer (light-painted trails) and a bloom post pass.
//
//  Scenarios (unchanged physics):
//      1  Magnetic dipole      (radiation-belt-like trapping)
//      2  Magnetic quadrupole  (focusing / hyperbolic field lines)
//      3  Cyclotron            (uniform B -> helical gyration)
//      4  Magnetic bottle      (magnetic mirror -> bouncing motion)
//      5  Electric dipole      (Coulomb E-field, accelerated test charges)
//      6  Electric quadrupole  (4 point charges, scattering)
//      7  E x B drift          (crossed uniform fields -> cycloid drift)
//      8  Penning trap         (uniform B + quadrupole E -> epicyclic orbits)
//
//  Stack: AppKit + MetalKit + simd + ImageIO. No third-party dependencies.
//  The Metal source is compiled at runtime (makeLibrary(source:)).
//
//  Build:
//      swiftc -O EMFieldDemo.swift -o EMFieldDemo \
//             -framework Cocoa -framework Metal -framework MetalKit
//
//  Controls:
//      1..8      switch scenario
//      Space     pause / resume (freezes the trails too)
//      R         reseed the swarm
//      F         toggle field lines        X   toggle axes
//      B         toggle bloom              H   toggle the HUD
//      A         camera auto-rotation      C   cycle particle colour mode
//      S         slow motion (0.25x)
//      [ / ]     halve / double the particle count
//      - / =     exposure down / up
//      , / .     trail persistence down / up
//      ; / '     point size down / up
//      0         reset all tunables
//      P         save a PNG of the current frame (to CWD)
//      O         save a 2x supersampled PNG
//      drag      orbit camera     scroll  zoom
// =============================================================================

import Cocoa
import Metal
import MetalKit
import simd
import ImageIO
import QuartzCore
import UniformTypeIdentifiers

// MARK: - Tunables

private var   gParticleCount = 200_000       // swarm size (mutable via [ ])
private let   kMaxParticles  = 2_000_000      // hard cap (buffer allocation)
private let   kReinject      = Float(5.5)     // respawn when |pos| exceeds this

/// Everything that can be tweaked live from the keyboard.
/// Press `0` to reset to these defaults.
struct Tunables {
    var pointSize:      Float  = 3.0    // particle sprite size (backing px)   ; / '
    var intensity:      Float  = 0.07   // per-particle additive brightness
    var fade:           Float  = 0.90   // trail persistence (0=none, 1=forever)  , / .
    var exposure:       Float  = 1.15   // tone-mapping exposure                  - / =
    var bloomStrength:  Float  = 1.35
    var bloomThreshold: Float  = 1.0
    var timeScale:      Float  = 1.0    // simulation speed multiplier            S
    var colorMode:      UInt32 = 0      // 0 speed · 1 local |field| · 2 ember    C
}

let kColorModeNames = ["speed", "|field|", "ember"]

// MARK: - CPU-side structs (must match the MSL structs below, byte-for-byte)

struct LineVertex {
    var position: SIMD3<Float>
    var color:    SIMD4<Float>
}

struct SimU {                 // compute uniforms (48 bytes)
    var dt: Float
    var qOverM: Float
    var reinject: Float
    var emitterRadius: Float
    var scenario: UInt32
    var substeps: UInt32
    var frameSeed: UInt32
    var count: UInt32
    var vmin: Float
    var vmax: Float
    var _p0: Float = 0
    var _p1: Float = 0
}

struct RenderU {              // particle vertex uniforms (96 bytes)
    var viewProj: float4x4
    var pointSize: Float
    var speedScale: Float
    var intensity: Float
    var colorMode: UInt32
    var scenario: UInt32      // lets the vertex shader sample fieldAt() for mode 1
    var _p0: Float = 0
    var _p1: Float = 0
    var _p2: Float = 0
}

struct LineU {               // line vertex uniforms (64 bytes)
    var viewProj: float4x4
}

struct CompositeU {          // composite fragment uniforms (16 bytes)
    var exposure: Float
    var bloomStrength: Float
    var bgIntensity: Float
    var _pad: Float = 0
}

@inline(__always) func rgba(_ c: SIMD3<Float>, _ a: Float) -> SIMD4<Float> {
    SIMD4<Float>(c.x, c.y, c.z, a)
}

// MARK: - Math helpers (Metal-compatible, NDC z in [0,1])

func perspectiveRH(fovyRadians: Float, aspect: Float, near: Float, far: Float) -> float4x4 {
    let ys = 1 / tan(fovyRadians * 0.5)
    let xs = ys / aspect
    let zs = far / (near - far)
    return float4x4(columns: (
        SIMD4<Float>(xs, 0,  0,         0),
        SIMD4<Float>(0,  ys, 0,         0),
        SIMD4<Float>(0,  0,  zs,       -1),
        SIMD4<Float>(0,  0,  zs * near, 0)
    ))
}

func lookAtRH(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> float4x4 {
    let z = normalize(eye - center)
    let x = normalize(cross(up, z))
    let y = cross(z, x)
    let t = SIMD3<Float>(-dot(x, eye), -dot(y, eye), -dot(z, eye))
    return float4x4(columns: (
        SIMD4<Float>(x.x, y.x, z.x, 0),
        SIMD4<Float>(x.y, y.y, z.y, 0),
        SIMD4<Float>(x.z, y.z, z.z, 0),
        SIMD4<Float>(t.x, t.y, t.z, 1)
    ))
}

/// Cool-to-warm 5-stop colour map, input clamped to [0,1] (CPU copy for lines).
func colormap(_ t: Float) -> SIMD3<Float> {
    let stops: [SIMD3<Float>] = [
        SIMD3(0.10, 0.20, 0.78),
        SIMD3(0.10, 0.74, 0.86),
        SIMD3(0.30, 0.85, 0.38),
        SIMD3(0.96, 0.85, 0.26),
        SIMD3(0.96, 0.30, 0.20)
    ]
    let x = max(0, min(1, t)) * 4.0
    let i = min(3, Int(x))
    return mix(stops[i], stops[i + 1], t: x - Float(i))
}

// MARK: - Scenario definition

enum LineField { case magnetic, electric }

struct Scenario {
    let name: String
    let field: (SIMD3<Float>) -> (E: SIMD3<Float>, B: SIMD3<Float>)
    let q: Float
    let m: Float
    let dt: Float
    let substeps: Int
    let cameraDistance: Float
    let seeds: [SIMD3<Float>]
    let uniformLineColor: SIMD3<Float>?
    // swarm emitter parameters
    var emitterRadius: Float = 2.4
    var vMin: Float = 0.6
    var vMax: Float = 1.8
    var speedScale: Float = 0.35
    // field-line tracing
    var lineField: LineField = .magnetic
    var stopSites: [SIMD3<Float>] = []
}

enum Scenarios {

    static let count = 8

    static func dipole() -> Scenario {
        let k: Float = 1.4
        let field: (SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) = { p in
            let r = max(length(p), 0.28)
            let r5 = r * r * r * r * r
            let x = p.x, y = p.y, z = p.z
            return (.zero, SIMD3(k * 3 * x * z / r5,
                                 k * 3 * y * z / r5,
                                 k * (3 * z * z - r * r) / r5))
        }
        var seeds: [SIMD3<Float>] = []
        for ai in 0..<6 {
            let az = Float(ai) * (.pi / 3)
            for theta in [Float(0.6), 1.0, 1.4, 1.9, 2.3] {
                let s = sin(theta), c = cos(theta)
                seeds.append(0.6 * SIMD3(s * cos(az), s * sin(az), c))
            }
        }
        return Scenario(
            name: "Magnetic dipole",
            field: field, q: 4.0, m: 1.0, dt: 0.0035, substeps: 8,
            cameraDistance: 7.0, seeds: seeds, uniformLineColor: nil,
            emitterRadius: 2.6, vMin: 0.8, vMax: 2.0, speedScale: 0.30)
    }

    static func quadrupole() -> Scenario {
        let g: Float = 2.0
        let field: (SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) = { p in
            (.zero, SIMD3(g * p.y, g * p.x, 0))
        }
        var seeds: [SIMD3<Float>] = []
        for radius in [Float(0.5), 1.0, 1.6, 2.2] {
            for ai in 0..<12 {
                let az = Float(ai) * (.pi / 6) + 0.13
                seeds.append(SIMD3(radius * cos(az), radius * sin(az), 0))
            }
        }
        return Scenario(
            name: "Magnetic quadrupole",
            field: field, q: 2.0, m: 1.0, dt: 0.004, substeps: 8,
            cameraDistance: 7.5, seeds: seeds, uniformLineColor: nil,
            emitterRadius: 2.4, vMin: 0.6, vMax: 1.6, speedScale: 0.30)
    }

    static func cyclotron() -> Scenario {
        let b0: Float = 2.5
        let field: (SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) = { _ in
            (.zero, SIMD3(0, 0, b0))
        }
        var seeds: [SIMD3<Float>] = []
        for ix in -2...2 {
            for iy in -2...2 {
                let p = SIMD3(Float(ix), Float(iy), 0)
                if length(p) <= 2.6 { seeds.append(p) }
            }
        }
        return Scenario(
            name: "Cyclotron (uniform B)",
            field: field, q: 1.0, m: 1.0, dt: 0.006, substeps: 6,
            cameraDistance: 7.0, seeds: seeds,
            uniformLineColor: SIMD3(0.25, 0.55, 0.95),
            emitterRadius: 2.6, vMin: 0.6, vMax: 1.8, speedScale: 0.35)
    }

    static func bottle() -> Scenario {
        let b0: Float = 1.0
        let L:  Float = 1.6
        let field: (SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) = { p in
            let invL2 = 1.0 / (L * L)
            return (.zero, SIMD3(-b0 * p.z * p.x * invL2,
                                 -b0 * p.z * p.y * invL2,
                                  b0 * (1 + p.z * p.z * invL2)))
        }
        var seeds: [SIMD3<Float>] = []
        for radius in [Float(0.4), 0.85, 1.3] {
            for ai in 0..<6 {
                let az = Float(ai) * (.pi / 3)
                seeds.append(SIMD3(radius * cos(az), radius * sin(az), 0))
            }
        }
        return Scenario(
            name: "Magnetic bottle (mirror)",
            field: field, q: 3.0, m: 1.0, dt: 0.0045, substeps: 8,
            cameraDistance: 7.0, seeds: seeds, uniformLineColor: nil,
            emitterRadius: 1.8, vMin: 0.6, vMax: 1.6, speedScale: 0.35)
    }

    static func coulomb(_ p: SIMD3<Float>, _ charges: [(SIMD3<Float>, Float)]) -> SIMD3<Float> {
        var e = SIMD3<Float>(0, 0, 0)
        for (c, qc) in charges {
            let d = p - c
            let r = max(length(d), 0.28)
            e += qc * d / (r * r * r)
        }
        return e
    }

    static func seedsAround(_ center: SIMD3<Float>, radius: Float,
                            azimuths: Int, polars: [Float]) -> [SIMD3<Float>] {
        var out: [SIMD3<Float>] = []
        for fi in 0..<azimuths {
            let phi = Float(fi) * (2 * .pi / Float(azimuths))
            for th in polars {
                let s = sin(th), c = cos(th)
                out.append(center + radius * SIMD3(c, s * cos(phi), s * sin(phi)))
            }
        }
        return out
    }

    static func electricDipole() -> Scenario {
        let charges: [(SIMD3<Float>, Float)] = [
            (SIMD3(-1.2, 0, 0),  1),
            (SIMD3( 1.2, 0, 0), -1)
        ]
        let field: (SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) = { p in
            (coulomb(p, charges), .zero)
        }
        let seeds = seedsAround(SIMD3(-1.2, 0, 0), radius: 0.4,
                                azimuths: 6, polars: [0.45, 0.9, 1.4, 1.9, 2.5])
        return Scenario(
            name: "Electric dipole",
            field: field, q: 1.0, m: 1.0, dt: 0.004, substeps: 8,
            cameraDistance: 7.5, seeds: seeds, uniformLineColor: nil,
            emitterRadius: 2.2, vMin: 0.4, vMax: 1.2, speedScale: 0.40,
            lineField: .electric, stopSites: charges.map { $0.0 })
    }

    static func electricQuadrupole() -> Scenario {
        let charges: [(SIMD3<Float>, Float)] = [
            (SIMD3( 1.2, 0, 0),  1),
            (SIMD3(-1.2, 0, 0),  1),
            (SIMD3(0,  1.2, 0), -1),
            (SIMD3(0, -1.2, 0), -1)
        ]
        let field: (SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) = { p in
            (coulomb(p, charges), .zero)
        }
        var seeds: [SIMD3<Float>] = []
        for plus in [SIMD3<Float>(1.2, 0, 0), SIMD3<Float>(-1.2, 0, 0)] {
            seeds += seedsAround(plus, radius: 0.4,
                                 azimuths: 8, polars: [0.5, 1.1, 1.7, 2.4])
        }
        return Scenario(
            name: "Electric quadrupole",
            field: field, q: 1.0, m: 1.0, dt: 0.004, substeps: 8,
            cameraDistance: 7.8, seeds: seeds, uniformLineColor: nil,
            emitterRadius: 2.4, vMin: 0.4, vMax: 1.2, speedScale: 0.40,
            lineField: .electric, stopSites: charges.map { $0.0 })
    }

    /// Crossed uniform fields: E along +x, B along +z.
    /// Every particle gyrates while its guiding centre drifts with
    /// v_d = E x B / |B|^2  (here: along -y), independent of charge and speed —
    /// the swarm turns into a river of cycloids all flowing the same way.
    static func crossedFields() -> Scenario {
        let e0: Float = 1.2
        let b0: Float = 2.0
        let field: (SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) = { _ in
            (SIMD3(e0, 0, 0), SIMD3(0, 0, b0))
        }
        // straight, uniform B lines on a grid (same style as the cyclotron)
        var seeds: [SIMD3<Float>] = []
        for ix in -2...2 {
            for iy in -2...2 {
                let p = SIMD3(Float(ix), Float(iy), 0)
                if length(p) <= 2.6 { seeds.append(p) }
            }
        }
        return Scenario(
            name: "E x B drift",
            field: field, q: 1.5, m: 1.0, dt: 0.005, substeps: 6,
            cameraDistance: 7.5, seeds: seeds,
            uniformLineColor: SIMD3(0.25, 0.55, 0.95),
            emitterRadius: 2.2, vMin: 0.4, vMax: 1.2, speedScale: 0.40)
    }

    /// Ideal Penning trap: strong uniform B_z plus the quadrupole field
    /// E = k (x, y, -2z), i.e. the potential  phi = k (z^2 - (x^2+y^2)/2).
    /// The E field confines axially (F_z = -2kq z) and *anti*-confines radially;
    /// B turns the radial escape into slow magnetron circulation. Stable while
    /// w_c^2 > 2 w_z^2  (here 9 > 2.4). Orbits are three superimposed motions:
    /// fast cyclotron + axial bounce + slow magnetron drift -> rosette trails.
    static func penning() -> Scenario {
        let b0: Float = 3.0
        let k:  Float = 0.6
        let field: (SIMD3<Float>) -> (SIMD3<Float>, SIMD3<Float>) = { p in
            (SIMD3(k * p.x, k * p.y, -2 * k * p.z), SIMD3(0, 0, b0))
        }
        // electric field lines of the quadrupole-of-revolution (hyperbola-like)
        var seeds: [SIMD3<Float>] = []
        for ai in 0..<10 {
            let az = Float(ai) * (.pi / 5)
            for z in [Float(-1.4), -0.7, 0.7, 1.4] {
                seeds.append(SIMD3(0.5 * cos(az), 0.5 * sin(az), z))
            }
        }
        return Scenario(
            name: "Penning trap",
            field: field, q: 1.0, m: 1.0, dt: 0.004, substeps: 8,
            cameraDistance: 6.0, seeds: seeds, uniformLineColor: nil,
            emitterRadius: 1.3, vMin: 0.2, vMax: 0.9, speedScale: 0.55,
            lineField: .electric)
    }

    static func make(_ index: Int) -> Scenario {
        switch index {
        case 0:  return dipole()
        case 1:  return quadrupole()
        case 2:  return cyclotron()
        case 3:  return bottle()
        case 4:  return electricDipole()
        case 5:  return electricQuadrupole()
        case 6:  return crossedFields()
        default: return penning()
        }
    }
}

// MARK: - Field-line tracing (CPU, RK4 along normalized field) — unchanged

func traceLine(seed: SIMD3<Float>, sign: Float, steps: Int, ds: Float,
               field: (SIMD3<Float>) -> SIMD3<Float>,
               sites: [SIMD3<Float>] = [], stopR: Float = 0) -> [SIMD3<Float>] {
    var pts: [SIMD3<Float>] = [seed]
    var p = seed
    func dir(_ x: SIMD3<Float>) -> SIMD3<Float> {
        let b = field(x)
        let n = length(b)
        return n > 1e-6 ? (sign * b / n) : SIMD3<Float>(0, 0, 0)
    }
    for _ in 0..<steps {
        let k1 = dir(p); if k1 == .zero { break }
        let k2 = dir(p + 0.5 * ds * k1)
        let k3 = dir(p + 0.5 * ds * k2)
        let k4 = dir(p + ds * k3)
        p += (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if length(p) > kReinject { break }
        if stopR > 0 {
            var hit = false
            for s in sites where length(p - s) < stopR { hit = true; break }
            if hit { pts.append(p); break }
        }
        pts.append(p)
    }
    return pts
}

func buildFieldLines(_ s: Scenario) -> [LineVertex] {
    let fvec: (SIMD3<Float>) -> SIMD3<Float> =
        (s.lineField == .electric) ? { s.field($0).E } : { s.field($0).B }
    let bothDirections = (s.lineField == .magnetic)

    var polylines: [[SIMD3<Float>]] = []
    for seed in s.seeds {
        let fwd = traceLine(seed: seed, sign: 1, steps: 420, ds: 0.03,
                            field: fvec, sites: s.stopSites, stopR: 0.32)
        if bothDirections {
            let bwd = traceLine(seed: seed, sign: -1, steps: 420, ds: 0.03,
                                field: fvec, sites: s.stopSites, stopR: 0.32)
            let line = Array(bwd.dropFirst().reversed()) + fwd
            if line.count > 1 { polylines.append(line) }
        } else if fwd.count > 1 {
            polylines.append(fwd)
        }
    }

    var lo = Float.greatestFiniteMagnitude
    var hi = -Float.greatestFiniteMagnitude
    for line in polylines {
        for p in line {
            let lb = log(length(fvec(p)) + 1e-6)
            lo = min(lo, lb); hi = max(hi, lb)
        }
    }
    let span = max(hi - lo, 1e-4)

    // Lines live in an HDR layer; push them above the bloom threshold so they glow.
    let lineBright: Float = 5
    func colorAt(_ p: SIMD3<Float>) -> SIMD4<Float> {
        if let c = s.uniformLineColor { return rgba(c * lineBright, 1.0) }
        let t = (log(length(fvec(p)) + 1e-6) - lo) / span
        return rgba(colormap(t) * lineBright, 1.0)
    }

    var verts: [LineVertex] = []
    for line in polylines {
        for i in 0..<(line.count - 1) {
            verts.append(LineVertex(position: line[i],     color: colorAt(line[i])))
            verts.append(LineVertex(position: line[i + 1], color: colorAt(line[i + 1])))
        }
    }
    return verts
}

func buildAxes() -> [LineVertex] {
    let L: Float = 3.0
    let a: Float = 0.30                       // additive glow amount
    func seg(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ c: SIMD3<Float>) -> [LineVertex] {
        [LineVertex(position: p0, color: rgba(c * a, 1.0)),
         LineVertex(position: p1, color: rgba(c * a, 1.0))]
    }
    return seg(SIMD3(-L,0,0), SIMD3(L,0,0), SIMD3(0.9,0.3,0.3))
         + seg(SIMD3(0,-L,0), SIMD3(0,L,0), SIMD3(0.3,0.9,0.3))
         + seg(SIMD3(0,0,-L), SIMD3(0,0,L), SIMD3(0.3,0.5,0.95))
}

// MARK: - World

final class World {
    private(set) var scenarioIndex = 0
    private(set) var scenario = Scenarios.make(0)

    var paused = false
    var showFieldLines = true
    var showAxes = true
    var bloomEnabled = true
    var autoRotate = true
    var hudVisible = true
    var tun = Tunables()

    var azimuth: Float = 0.7
    var elevation: Float = 0.4
    var distance: Float = 7.0

    private(set) var fieldLineVerts: [LineVertex] = []
    private(set) var axisVerts: [LineVertex] = buildAxes()

    init() { loadScenario(0) }

    func loadScenario(_ i: Int) {
        scenarioIndex = ((i % Scenarios.count) + Scenarios.count) % Scenarios.count
        scenario = Scenarios.make(scenarioIndex)
        distance = scenario.cameraDistance
        fieldLineVerts = buildFieldLines(scenario)
    }
}

// MARK: - Metal shader source (compiled at runtime)

private let kShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct SimU {
    float dt, qOverM, reinject, emitterRadius;
    uint  scenario, substeps, frameSeed, count;
    float vmin, vmax, p0, p1;
};

struct RenderU {
    float4x4 viewProj;
    float pointSize;
    float speedScale;
    float intensity;
    uint  colorMode;
    uint  scenario;
    float p0, p1, p2;
};

struct LineU   { float4x4 viewProj; };
struct CompU   { float exposure; float bloomStrength; float bgIntensity; float pad; };

constexpr sampler samp(coord::normalized, address::clamp_to_edge, filter::linear);

// ---- field model -----------------------------------------------------------
struct Field { float3 E; float3 B; };

inline Field fieldAt(float3 p, uint sc) {
    Field f; f.E = float3(0.0); f.B = float3(0.0);
    if (sc == 0u) {                         // magnetic dipole
        float k = 1.4;
        float r = max(length(p), 0.28);
        float r5 = r*r*r*r*r;
        f.B = float3(k*3.0*p.x*p.z/r5, k*3.0*p.y*p.z/r5, k*(3.0*p.z*p.z - r*r)/r5);
    } else if (sc == 1u) {                  // magnetic quadrupole
        float g = 2.0; f.B = float3(g*p.y, g*p.x, 0.0);
    } else if (sc == 2u) {                  // cyclotron
        f.B = float3(0.0, 0.0, 2.5);
    } else if (sc == 3u) {                  // magnetic bottle
        float b0 = 1.0, L = 1.6, invL2 = 1.0/(L*L);
        f.B = float3(-b0*p.z*p.x*invL2, -b0*p.z*p.y*invL2, b0*(1.0 + p.z*p.z*invL2));
    } else if (sc == 4u) {                  // electric dipole
        float3 e = float3(0.0);
        float3 d0 = p - float3(-1.2,0,0); float r0 = max(length(d0),0.28); e +=  d0/(r0*r0*r0);
        float3 d1 = p - float3( 1.2,0,0); float r1 = max(length(d1),0.28); e += -d1/(r1*r1*r1);
        f.E = e;
    } else if (sc == 6u) {                  // E x B drift (crossed uniform fields)
        f.E = float3(1.2, 0.0, 0.0);
        f.B = float3(0.0, 0.0, 2.0);
    } else if (sc == 7u) {                  // Penning trap
        float k = 0.6;
        f.E = float3(k*p.x, k*p.y, -2.0*k*p.z);
        f.B = float3(0.0, 0.0, 3.0);
    } else {                                // electric quadrupole
        float3 e = float3(0.0);
        float3 ch[4] = { float3(1.2,0,0), float3(-1.2,0,0), float3(0,1.2,0), float3(0,-1.2,0) };
        float  qs[4] = { 1.0, 1.0, -1.0, -1.0 };
        for (int i = 0; i < 4; ++i) {
            float3 d = p - ch[i]; float r = max(length(d),0.28); e += qs[i]*d/(r*r*r);
        }
        f.E = e;
    }
    return f;
}

// ---- rng + emitter ----------------------------------------------------------
inline uint  pcg(thread uint& s) {
    s = s*747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
inline float rnd(thread uint& s) { return float(pcg(s)) * (1.0/4294967296.0); }

inline float3 sampleBall(thread uint& s, float R) {
    float u1 = rnd(s), u2 = rnd(s), u3 = rnd(s);
    float r  = R * pow(u1, 1.0/3.0);
    float ct = 2.0*u2 - 1.0;
    float st = sqrt(max(0.0, 1.0 - ct*ct));
    float ph = 6.28318530718 * u3;
    return r * float3(st*cos(ph), st*sin(ph), ct);
}
inline float3 sampleVel(thread uint& s, float vmin, float vmax) {
    float u1 = rnd(s), u2 = rnd(s), u3 = rnd(s);
    float sp = mix(vmin, vmax, u1);
    float ct = 2.0*u2 - 1.0;
    float st = sqrt(max(0.0, 1.0 - ct*ct));
    float ph = 6.28318530718 * u3;
    return sp * float3(st*cos(ph), st*sin(ph), ct);
}

// ---- Boris pusher (compute) -------------------------------------------------
kernel void integrate(device float4* posBuf  [[buffer(0)]],
                      device float4* velBuf  [[buffer(1)]],
                      constant SimU& u       [[buffer(2)]],
                      uint gid               [[thread_position_in_grid]]) {
    if (gid >= u.count) return;
    float3 pos = posBuf[gid].xyz;
    float3 vel = velBuf[gid].xyz;
    float  h   = u.qOverM * u.dt * 0.5;

    for (uint i = 0; i < u.substeps; ++i) {
        Field f = fieldAt(pos, u.scenario);
        float3 vm = vel + h * f.E;
        float3 t  = h * f.B;
        float3 s  = (2.0 / (1.0 + dot(t,t))) * t;
        float3 vp = vm + cross(vm, t);
        vel = vm + cross(vp, s) + h * f.E;
        pos = pos + vel * u.dt;
        if (length(pos) > u.reinject) {
            uint st = gid*9781u + u.frameSeed*6271u + i*1300097u + 1u;
            pos = sampleBall(st, u.emitterRadius);
            vel = sampleVel(st, u.vmin, u.vmax);
        }
    }
    posBuf[gid] = float4(pos, length(vel));   // pack speed into .w for the renderer
    velBuf[gid] = float4(vel, 0.0);
}

// ---- particle sprites -------------------------------------------------------
struct POut { float4 position [[position]]; float point_size [[point_size]]; float4 color; };

inline float3 colormap(float t) {
    const float3 c0 = float3(0.10, 0.20, 0.78);
    const float3 c1 = float3(0.10, 0.74, 0.86);
    const float3 c2 = float3(0.30, 0.85, 0.38);
    const float3 c3 = float3(0.96, 0.85, 0.26);
    const float3 c4 = float3(0.96, 0.30, 0.20);
    float x = clamp(t, 0.0, 1.0) * 4.0;
    if (x < 1.0) return mix(c0, c1, x);
    if (x < 2.0) return mix(c1, c2, x-1.0);
    if (x < 3.0) return mix(c2, c3, x-2.0);
    return mix(c3, c4, x-3.0);
}

vertex POut particle_vertex(const device float4* posBuf [[buffer(0)]],
                            constant RenderU& u          [[buffer(1)]],
                            uint vid                     [[vertex_id]]) {
    float4 pw = posBuf[vid];
    POut o;
    o.position   = u.viewProj * float4(pw.xyz, 1.0);
    o.point_size = u.pointSize;

    float3 c;
    if (u.colorMode == 1u) {
        // colour by local field magnitude at the particle position
        Field f = fieldAt(pw.xyz, u.scenario);
        float m = length(f.B) + length(f.E);
        c = colormap(clamp(log(1.0 + m) * 0.55, 0.0, 1.0));
    } else if (u.colorMode == 2u) {
        // monochrome "ember": dark red embers -> white-hot, weighted by speed
        float t = clamp(pw.w * u.speedScale, 0.0, 1.0);
        c = mix(float3(0.30, 0.09, 0.03), float3(1.0, 0.85, 0.55), t) * (0.4 + 1.6*t);
    } else {
        // default: colour by speed
        c = colormap(clamp(pw.w * u.speedScale, 0.0, 1.0));
    }
    o.color = float4(c * u.intensity, 1.0);
    return o;
}

fragment float4 particle_fragment(POut in [[stage_in]], float2 pc [[point_coord]]) {
    float2 d = pc - 0.5;
    float a = exp(-dot(d, d) * 12.0);        // soft gaussian dot
    return float4(in.color.rgb * a, a);      // additive (premultiplied)
}

// ---- fullscreen helpers -----------------------------------------------------
struct FSOut { float4 position [[position]]; float2 uv; };

vertex FSOut fs_vertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);   // (0,0)(2,0)(0,2)
    FSOut o;
    o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(p.x, 1.0 - p.y);                // upright sampling
    return o;
}

// fade pass: fragment output is ignored (source factor = zero); dst *= blendColor.
fragment float4 fade_fragment(FSOut in [[stage_in]]) { return float4(0.0); }

// bloom bright-pass (half res) — sums the particle accumulator and the line layer
fragment float4 bright_fragment(FSOut in [[stage_in]],
                                texture2d<float> accum [[texture(0)]],
                                texture2d<float> lines [[texture(1)]],
                                constant float& threshold [[buffer(0)]]) {
    float3 c = accum.sample(samp, in.uv).rgb + lines.sample(samp, in.uv).rgb;
    float  l = max(max(c.r, c.g), c.b);
    float  k = max(0.0, l - threshold);
    float3 outc = (l > 1e-5) ? c * (k / l) : float3(0.0);
    return float4(outc, 1.0);
}

// separable gaussian blur (9-tap)
fragment float4 blur_fragment(FSOut in [[stage_in]],
                              texture2d<float> src [[texture(0)]],
                              constant float2& dir [[buffer(0)]]) {
    const float w[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };
    float3 acc = src.sample(samp, in.uv).rgb * w[0];
    for (int i = 1; i < 5; ++i) {
        float2 off = dir * float(i);
        acc += src.sample(samp, in.uv + off).rgb * w[i];
        acc += src.sample(samp, in.uv - off).rgb * w[i];
    }
    return float4(acc, 1.0);
}

inline float3 aces(float3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a*x + b)) / (x * (c*x + d) + e), 0.0, 1.0);
}

fragment float4 composite_fragment(FSOut in [[stage_in]],
                                   texture2d<float> accum [[texture(0)]],
                                   texture2d<float> bloom [[texture(1)]],
                                   texture2d<float> lines [[texture(2)]],
                                   constant CompU& u      [[buffer(0)]]) {
    float3 hdr = accum.sample(samp, in.uv).rgb
               + lines.sample(samp, in.uv).rgb
               + u.bloomStrength * bloom.sample(samp, in.uv).rgb;
    float3 mapped = aces(u.exposure * hdr);
    // dark radial-ish gradient background
    float2 q = in.uv - 0.5;
    float  vig = 1.0 - 0.8 * dot(q, q);
    float3 bg = mix(float3(0.015, 0.02, 0.05), float3(0.0, 0.0, 0.015), in.uv.y) * vig * u.bgIntensity;
    return float4(mapped + bg, 1.0);
}

// ---- field lines / axes -----------------------------------------------------
struct LineVtx { float3 position; float4 color; };
struct LineOut { float4 position [[position]]; float4 color; };

vertex LineOut line_vertex(const device LineVtx* v [[buffer(0)]],
                           constant LineU& u        [[buffer(1)]],
                           uint vid                 [[vertex_id]]) {
    LineOut o;
    o.position = u.viewProj * float4(v[vid].position, 1.0);
    o.color    = v[vid].color;
    return o;
}
fragment float4 line_fragment(LineOut in [[stage_in]]) { return in.color; }
"""

// MARK: - Renderer

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let world: World

    // pipelines
    private var integratePSO: MTLComputePipelineState!
    private var particlePSO:  MTLRenderPipelineState!
    private var fadePSO:      MTLRenderPipelineState!
    private var brightPSO:    MTLRenderPipelineState!
    private var blurPSO:      MTLRenderPipelineState!
    private var compositePSO: MTLRenderPipelineState!
    private var linePSO:      MTLRenderPipelineState!
    // separate composite/line pipelines for PNG capture share the same bgra8 format,
    // so they are reused directly.

    // particle state
    private var posBuf: MTLBuffer!
    private var velBuf: MTLBuffer!

    // cached line geometry (rebuilt on scenario change, not per frame)
    private var fieldBuffer: MTLBuffer?
    private var axisBuffer:  MTLBuffer?

    // offscreen targets
    private var accumTex: MTLTexture?
    private var lineTex:  MTLTexture?     // full res, redrawn fresh each frame
    private var bloomA:   MTLTexture?     // half res
    private var bloomB:   MTLTexture?
    private var pxW = 0, pxH = 0
    private var needsAccumClear = true

    private var frameSeed: UInt32 = 1
    private var captureRequested = false
    private var captureScale = 1

    // fps (exponential moving average) + throttled HUD refresh
    private(set) var fps: Double = 0
    private var lastFrameTime: CFTimeInterval = 0
    private var lastHUDTime:   CFTimeInterval = 0
    var onHUD: ((String) -> Void)?

    private let drawableFormat: MTLPixelFormat = .bgra8Unorm
    private let hdrFormat:       MTLPixelFormat = .rgba16Float

    init?(view: MTKView, world: World) {
        guard let dev = view.device ?? MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else { return nil }
        self.device = dev; self.queue = q; self.world = world
        super.init()

        view.colorPixelFormat = drawableFormat
        view.depthStencilPixelFormat = .invalid           // no depth needed
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        do { try buildPipelines() } catch {
            print("Pipeline build failed: \(error)"); return nil
        }
        allocParticles()
        seedParticles()
        refreshLineBuffers()
    }

    func refreshLineBuffers() {
        if !world.axisVerts.isEmpty {
            axisBuffer = device.makeBuffer(
                bytes: world.axisVerts,
                length: MemoryLayout<LineVertex>.stride * world.axisVerts.count,
                options: .storageModeShared)
        }
        if !world.fieldLineVerts.isEmpty {
            fieldBuffer = device.makeBuffer(
                bytes: world.fieldLineVerts,
                length: MemoryLayout<LineVertex>.stride * world.fieldLineVerts.count,
                options: .storageModeShared)
        } else {
            fieldBuffer = nil
        }
    }

    // ---- pipelines ----
    private func buildPipelines() throws {
        let lib = try device.makeLibrary(source: kShaderSource, options: nil)

        integratePSO = try device.makeComputePipelineState(
            function: lib.makeFunction(name: "integrate")!)

        func render(_ vfn: String, _ ffn: String, format: MTLPixelFormat,
                    blend: String) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction   = lib.makeFunction(name: vfn)
            d.fragmentFunction = lib.makeFunction(name: ffn)
            let ca = d.colorAttachments[0]!
            ca.pixelFormat = format
            switch blend {
            case "additive":
                ca.isBlendingEnabled = true
                ca.rgbBlendOperation = .add; ca.alphaBlendOperation = .add
                ca.sourceRGBBlendFactor = .one;  ca.sourceAlphaBlendFactor = .one
                ca.destinationRGBBlendFactor = .one; ca.destinationAlphaBlendFactor = .one
            case "fade":  // dst *= blendColor   (source factor zero)
                ca.isBlendingEnabled = true
                ca.rgbBlendOperation = .add; ca.alphaBlendOperation = .add
                ca.sourceRGBBlendFactor = .zero; ca.sourceAlphaBlendFactor = .zero
                ca.destinationRGBBlendFactor = .blendColor
                ca.destinationAlphaBlendFactor = .blendAlpha
            case "alpha":
                ca.isBlendingEnabled = true
                ca.rgbBlendOperation = .add; ca.alphaBlendOperation = .add
                ca.sourceRGBBlendFactor = .sourceAlpha
                ca.sourceAlphaBlendFactor = .one
                ca.destinationRGBBlendFactor = .oneMinusSourceAlpha
                ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            default:      // replace
                ca.isBlendingEnabled = false
            }
            return try device.makeRenderPipelineState(descriptor: d)
        }

        particlePSO  = try render("particle_vertex", "particle_fragment", format: hdrFormat,      blend: "additive")
        fadePSO      = try render("fs_vertex",       "fade_fragment",     format: hdrFormat,      blend: "fade")
        brightPSO    = try render("fs_vertex",       "bright_fragment",   format: hdrFormat,      blend: "none")
        blurPSO      = try render("fs_vertex",       "blur_fragment",     format: hdrFormat,      blend: "none")
        compositePSO = try render("fs_vertex",       "composite_fragment",format: drawableFormat, blend: "none")
        linePSO      = try render("line_vertex",     "line_fragment",     format: hdrFormat,      blend: "additive")
    }

    // ---- particles ----
    private func allocParticles() {
        let len = MemoryLayout<SIMD4<Float>>.stride * kMaxParticles
        posBuf = device.makeBuffer(length: len, options: .storageModeShared)
        velBuf = device.makeBuffer(length: len, options: .storageModeShared)
    }

    /// Wipe the light-painted trails (used on reseed / scenario switch so stale
    /// trails from the previous field don't linger under the new one).
    func clearTrails() { needsAccumClear = true }

    func seedParticles() {
        clearTrails()
        let s = world.scenario
        let pos = posBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: kMaxParticles)
        let vel = velBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: kMaxParticles)
        for i in 0..<gParticleCount {
            // uniform point in ball
            let u1 = Float.random(in: 0...1), u2 = Float.random(in: 0...1), u3 = Float.random(in: 0...1)
            let r = s.emitterRadius * pow(u1, 1.0/3.0)
            let ct = 2*u2 - 1, st = (max(0, 1 - ct*ct)).squareRoot(), ph = 2*Float.pi*u3
            let p = r * SIMD3<Float>(st*cos(ph), st*sin(ph), ct)
            // random velocity
            let v1 = Float.random(in: 0...1), v2 = Float.random(in: 0...1), v3 = Float.random(in: 0...1)
            let sp = s.vMin + (s.vMax - s.vMin) * v1
            let ct2 = 2*v2 - 1, st2 = (max(0, 1 - ct2*ct2)).squareRoot(), ph2 = 2*Float.pi*v3
            let v = sp * SIMD3<Float>(st2*cos(ph2), st2*sin(ph2), ct2)
            pos[i] = SIMD4<Float>(p, length(v))
            vel[i] = SIMD4<Float>(v, 0)
        }
    }

    // ---- offscreen targets ----
    private func resize(_ w: Int, _ h: Int) {
        guard w > 0, h > 0 else { return }
        pxW = w; pxH = h
        func tex(_ tw: Int, _ th: Int) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: hdrFormat, width: max(1, tw), height: max(1, th), mipmapped: false)
            d.usage = [.renderTarget, .shaderRead]; d.storageMode = .private
            return device.makeTexture(descriptor: d)!
        }
        accumTex = tex(w, h)
        lineTex  = tex(w, h)
        bloomA = tex(w/2, h/2)
        bloomB = tex(w/2, h/2)
        needsAccumClear = true
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        resize(Int(size.width), Int(size.height))
    }

    // ---- camera ----
    private func viewProj(aspect: Float) -> float4x4 {
        let el = max(-1.45, min(1.45, world.elevation))
        let az = world.azimuth
        let eye = SIMD3<Float>(world.distance * cos(el) * cos(az),
                               world.distance * sin(el),
                               world.distance * cos(el) * sin(az))
        let viewM = lookAtRH(eye: eye, center: .zero, up: SIMD3<Float>(0,1,0))
        let projM = perspectiveRH(fovyRadians: 0.9, aspect: aspect, near: 0.05, far: 100)
        return projM * viewM
    }

    // ---- per-frame passes (shared by screen + capture composite) ----

    private func simStep(_ cmd: MTLCommandBuffer) {
        guard !world.paused else { return }
        guard let enc = cmd.makeComputeCommandEncoder() else { return }
        var u = SimU(dt: world.scenario.dt * world.tun.timeScale,
                     qOverM: world.scenario.q / world.scenario.m,
                     reinject: kReinject,
                     emitterRadius: world.scenario.emitterRadius,
                     scenario: UInt32(world.scenarioIndex),
                     substeps: UInt32(world.scenario.substeps),
                     frameSeed: frameSeed,
                     count: UInt32(gParticleCount),
                     vmin: world.scenario.vMin, vmax: world.scenario.vMax)
        enc.setComputePipelineState(integratePSO)
        enc.setBuffer(posBuf, offset: 0, index: 0)
        enc.setBuffer(velBuf, offset: 0, index: 1)
        enc.setBytes(&u, length: MemoryLayout<SimU>.stride, index: 2)
        let tg = MTLSize(width: 256, height: 1, depth: 1)
        let ng = MTLSize(width: (gParticleCount + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(ng, threadsPerThreadgroup: tg)
        enc.endEncoding()
    }

    private func accumStep(_ cmd: MTLCommandBuffer, vp: float4x4) {
        guard let accum = accumTex else { return }
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = accum
        rpd.colorAttachments[0].loadAction = needsAccumClear ? .clear : .load
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        rpd.colorAttachments[0].storeAction = .store
        needsAccumClear = false
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }

        // 1) fade the accumulated trails: dst *= fade
        if !world.paused {
            let fade = world.tun.fade
            enc.setRenderPipelineState(fadePSO)
            enc.setBlendColor(red: fade, green: fade, blue: fade, alpha: fade)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

            // 2) additive particle sprites
            var ru = RenderU(viewProj: vp, pointSize: world.tun.pointSize,
                             speedScale: world.scenario.speedScale,
                             intensity: world.tun.intensity,
                             colorMode: world.tun.colorMode,
                             scenario: UInt32(world.scenarioIndex))
            enc.setRenderPipelineState(particlePSO)
            enc.setVertexBuffer(posBuf, offset: 0, index: 0)
            enc.setVertexBytes(&ru, length: MemoryLayout<RenderU>.stride, index: 1)
            enc.drawPrimitives(type: .point, vertexStart: 0, vertexCount: gParticleCount)
        }
        enc.endEncoding()
    }

    private func bloomStep(_ cmd: MTLCommandBuffer) {
        guard world.bloomEnabled,
              let accum = accumTex, let lines = lineTex,
              let bA = bloomA, let bB = bloomB else { return }

        func pass(_ target: MTLTexture, _ src: MTLTexture, src2: MTLTexture? = nil,
                  pso: MTLRenderPipelineState,
                  setup: (MTLRenderCommandEncoder) -> Void) {
            let rpd = MTLRenderPassDescriptor()
            rpd.colorAttachments[0].texture = target
            rpd.colorAttachments[0].loadAction = .dontCare
            rpd.colorAttachments[0].storeAction = .store
            guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
            enc.setRenderPipelineState(pso)
            enc.setFragmentTexture(src, index: 0)
            if let s2 = src2 { enc.setFragmentTexture(s2, index: 1) }
            setup(enc)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }

        // bright pass: accum + lines -> bloomA
        pass(bA, accum, src2: lines, pso: brightPSO) { enc in
            var thr = world.tun.bloomThreshold
            enc.setFragmentBytes(&thr, length: MemoryLayout<Float>.stride, index: 0)
        }
        let txl = SIMD2<Float>(1.0 / Float(max(1, pxW/2)), 1.0 / Float(max(1, pxH/2)))
        // blur H: bloomA -> bloomB
        pass(bB, bA, pso: blurPSO) { enc in
            var dir = SIMD2<Float>(txl.x, 0)
            enc.setFragmentBytes(&dir, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        }
        // blur V: bloomB -> bloomA
        pass(bA, bB, pso: blurPSO) { enc in
            var dir = SIMD2<Float>(0, txl.y)
            enc.setFragmentBytes(&dir, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
        }
    }

    // Draw field lines + axes fresh into the HDR line layer (no persistence -> crisp,
    // no smear under rotation; bloom then turns thin lines into glowing curves).
    private func lineStep(_ cmd: MTLCommandBuffer, vp: float4x4) {
        guard let lt = lineTex else { return }
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = lt
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        rpd.colorAttachments[0].storeAction = .store
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        enc.setRenderPipelineState(linePSO)
        var lu = LineU(viewProj: vp)
        enc.setVertexBytes(&lu, length: MemoryLayout<LineU>.stride, index: 1)
        if world.showAxes, let ab = axisBuffer {
            enc.setVertexBuffer(ab, offset: 0, index: 0)
            enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: world.axisVerts.count)
        }
        if world.showFieldLines, let fb = fieldBuffer {
            enc.setVertexBuffer(fb, offset: 0, index: 0)
            enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: world.fieldLineVerts.count)
        }
        enc.endEncoding()
    }

    private func compositeStep(_ enc: MTLRenderCommandEncoder, vp: float4x4) {
        guard let accum = accumTex, let lines = lineTex else { return }
        enc.setRenderPipelineState(compositePSO)
        enc.setFragmentTexture(accum, index: 0)
        enc.setFragmentTexture(world.bloomEnabled ? (bloomA ?? accum) : accum, index: 1)
        enc.setFragmentTexture(lines, index: 2)
        var cu = CompositeU(exposure: world.tun.exposure,
                            bloomStrength: world.bloomEnabled ? world.tun.bloomStrength : 0,
                            bgIntensity: 1.0)
        enc.setFragmentBytes(&cu, length: MemoryLayout<CompositeU>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }

    // ---- MTKViewDelegate ----
    func draw(in view: MTKView) {
        let w = Int(view.drawableSize.width), h = Int(view.drawableSize.height)
        if accumTex == nil || pxW != w || pxH != h { resize(w, h) }
        let aspect = Float(max(w, 1)) / Float(max(h, 1))
        let vp = viewProj(aspect: aspect)

        guard let cmd = queue.makeCommandBuffer() else { return }

        simStep(cmd)
        accumStep(cmd, vp: vp)
        lineStep(cmd, vp: vp)
        bloomStep(cmd)

        if let rpd = view.currentRenderPassDescriptor,
           let drawable = view.currentDrawable,
           let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) {
            compositeStep(enc, vp: vp)
            enc.endEncoding()
            cmd.present(drawable)
        }
        cmd.commit()

        if captureRequested { captureRequested = false; capture(vp: vp, scale: captureScale) }

        if world.autoRotate && !world.paused { world.azimuth += 0.0025 }
        frameSeed &+= 1

        // fps (EMA) + throttled HUD refresh (draw runs on the main thread)
        let now = CACurrentMediaTime()
        if lastFrameTime > 0 {
            let inst = 1.0 / max(now - lastFrameTime, 1e-6)
            fps = fps == 0 ? inst : fps * 0.95 + inst * 0.05
        }
        lastFrameTime = now
        if now - lastHUDTime > 0.25 {
            lastHUDTime = now
            onHUD?(hudText())
        }
    }

    private func hudText() -> String {
        let t = world.tun
        let k = gParticleCount >= 1000 ? "\(gParticleCount / 1000)k" : "\(gParticleCount)"
        let flags = [
            world.bloomEnabled   ? "bloom" : nil,
            world.showFieldLines ? "lines" : nil,
            world.showAxes       ? "axes"  : nil,
            world.autoRotate     ? "spin"  : nil,
            world.paused         ? "PAUSED" : nil
        ].compactMap { $0 }.joined(separator: " · ")
        func f(_ v: Float, _ digits: Int = 2) -> String { String(format: "%.\(digits)f", v) }
        let mode = kColorModeNames[Int(t.colorMode) % kColorModeNames.count]
        return "[\(world.scenarioIndex + 1)/\(Scenarios.count)] \(world.scenario.name)"
             + "   \(k) particles   \(Int(fps.rounded())) fps\n"
             + "color \(mode) (C)   exposure \(f(t.exposure)) (-/=)   trail \(f(t.fade, 3)) (,/.)"
             + "   size \(f(t.pointSize, 1)) (;/')   time \(f(t.timeScale))x (S)\n"
             + flags
    }

    func requestCapture(scale: Int = 1) {
        captureScale = max(1, min(4, scale))
        captureRequested = true
    }

    // ---- PNG capture (composite into an offscreen bgra8 target) ----
    // scale > 1 re-composites the HDR layers into a larger target (upsampled
    // trails, but crisp tone mapping / gradient) — handy for wallpapers.
    private func capture(vp: float4x4, scale: Int = 1) {
        let W = max(1, pxW * scale), H = max(1, pxH * scale)
        let cd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: drawableFormat, width: W, height: H, mipmapped: false)
        cd.usage = [.renderTarget, .shaderRead]; cd.storageMode = .private
        guard let colorTex = device.makeTexture(descriptor: cd),
              let cmd = queue.makeCommandBuffer() else { return }

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = colorTex
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        rpd.colorAttachments[0].storeAction = .store
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        compositeStep(enc, vp: vp)
        enc.endEncoding()

        let rowBytes = ((W * 4 + 255) / 256) * 256
        guard let outBuf = device.makeBuffer(length: rowBytes * H, options: .storageModeShared),
              let blit = cmd.makeBlitCommandEncoder() else { return }
        blit.copy(from: colorTex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: W, height: H, depth: 1),
                  to: outBuf, destinationOffset: 0,
                  destinationBytesPerRow: rowBytes,
                  destinationBytesPerImage: rowBytes * H)
        blit.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        writePNG(outBuf.contents(), width: W, height: H, rowBytes: rowBytes)
    }

    private func writePNG(_ data: UnsafeMutableRawPointer, width: Int, height: Int, rowBytes: Int) {
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { return }
        let bmp = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(data: data, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: rowBytes,
                                  space: cs, bitmapInfo: bmp),
              let img = ctx.makeImage() else { return }
        let name = "emfield_\(Int(Date().timeIntervalSince1970)).png"
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(name)
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL,
                                                         UTType.png.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(dest, img, nil)
        if CGImageDestinationFinalize(dest) { print("Saved \(url.path)") }
        else { print("PNG export failed") }
    }
}

// MARK: - View (input)

final class DemoView: MTKView {
    weak var world: World?
    var onScenarioChange: (() -> Void)?
    var onReseed: (() -> Void)?
    var onCapture: (() -> Void)?

    override var acceptsFirstResponder: Bool { true }

    var onCaptureHiRes: (() -> Void)?
    var onHUDToggle: (() -> Void)?

    override func keyDown(with event: NSEvent) {
        guard let w = world else { return }
        switch event.charactersIgnoringModifiers ?? "" {
        case "1": switchTo(0)
        case "2": switchTo(1)
        case "3": switchTo(2)
        case "4": switchTo(3)
        case "5": switchTo(4)
        case "6": switchTo(5)
        case "7": switchTo(6)
        case "8": switchTo(7)
        case " ": w.paused.toggle()
        case "r", "R": onReseed?()
        case "f", "F": w.showFieldLines.toggle()
        case "x", "X": w.showAxes.toggle()
        case "b", "B": w.bloomEnabled.toggle()
        case "a", "A": w.autoRotate.toggle()
        case "h", "H": w.hudVisible.toggle(); onHUDToggle?()
        case "c", "C": w.tun.colorMode = (w.tun.colorMode + 1) % 3
        case "s", "S": w.tun.timeScale = (w.tun.timeScale < 1.0) ? 1.0 : 0.25
        case "-": w.tun.exposure  = max(0.1, w.tun.exposure - 0.1)
        case "=": w.tun.exposure  = min(6.0, w.tun.exposure + 0.1)
        case ",": w.tun.fade      = max(0.0, w.tun.fade - 0.01)
        case ".": w.tun.fade      = min(0.995, w.tun.fade + 0.01)
        case ";": w.tun.pointSize = max(1.0, w.tun.pointSize - 0.5)
        case "'": w.tun.pointSize = min(12.0, w.tun.pointSize + 0.5)
        case "0": w.tun = Tunables()
        case "[":
            gParticleCount = max(10_000, gParticleCount / 2); onReseed?(); updateTitle()
        case "]":
            gParticleCount = min(kMaxParticles, gParticleCount * 2); onReseed?(); updateTitle()
        case "p", "P": onCapture?()
        case "o", "O": onCaptureHiRes?()
        default: break
        }
    }

    private func switchTo(_ i: Int) {
        world?.loadScenario(i)
        onScenarioChange?()
        updateTitle()
    }

    override func mouseDragged(with event: NSEvent) {
        guard let w = world else { return }
        w.azimuth   -= Float(event.deltaX) * 0.01
        w.elevation += Float(event.deltaY) * 0.01
    }

    override func scrollWheel(with event: NSEvent) {
        guard let w = world else { return }
        w.distance *= Float(1.0 - event.deltaY * 0.02)
        w.distance = max(2.5, min(20.0, w.distance))
    }

    func updateTitle() {
        let k = gParticleCount >= 1000 ? "\(gParticleCount/1000)k" : "\(gParticleCount)"
        window?.title = "EM Field Demo  —  [\(world?.scenario.name ?? "")]  ·  \(k) particles"
            + "   (1-8 · H help/HUD · P png)"
    }
}

// MARK: - App bootstrap

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var renderer: Renderer!
    var hud: NSTextField!
    let world = World()

    func applicationDidFinishLaunching(_ notification: Notification) {
        let frame = NSRect(x: 0, y: 0, width: 1100, height: 760)
        window = NSWindow(contentRect: frame,
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered, defer: false)
        window.center()

        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this machine.")
        }
        let view = DemoView(frame: frame, device: device)
        view.world = world
        view.preferredFramesPerSecond = 60

        guard let r = Renderer(view: view, world: world) else {
            fatalError("Failed to create the Metal renderer.")
        }
        renderer = r
        view.delegate = r
        view.onScenarioChange = { [weak r] in r?.seedParticles(); r?.refreshLineBuffers() }
        view.onReseed         = { [weak r] in r?.seedParticles() }
        view.onCapture        = { [weak r] in r?.requestCapture(scale: 1) }
        view.onCaptureHiRes   = { [weak r] in r?.requestCapture(scale: 2) }

        // HUD overlay (top-left), refreshed ~4x per second from the render loop
        hud = NSTextField(labelWithString: "")
        hud.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        hud.textColor = NSColor(white: 0.88, alpha: 0.95)
        hud.backgroundColor = NSColor(white: 0, alpha: 0.35)
        hud.drawsBackground = true
        hud.usesSingleLineMode = false
        hud.maximumNumberOfLines = 0
        hud.lineBreakMode = .byClipping
        hud.autoresizingMask = [.minYMargin]     // stick to the top on resize
        view.addSubview(hud)

        view.onHUDToggle = { [weak self] in self?.hud.isHidden = !(self?.world.hudVisible ?? true) }
        r.onHUD = { [weak self] text in
            guard let self, self.world.hudVisible else { return }
            self.hud.stringValue = text
            self.hud.sizeToFit()
            let pad: CGFloat = 10
            let b = self.window.contentView?.bounds ?? .zero
            self.hud.setFrameOrigin(NSPoint(x: pad, y: b.height - self.hud.frame.height - pad))
        }

        window.contentView = view
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        view.updateTitle()
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }
}

func installMenu() {
    let mainMenu = NSMenu()
    let appItem = NSMenuItem()
    mainMenu.addItem(appItem)
    let appMenu = NSMenu()
    appMenu.addItem(withTitle: "Quit", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
    appItem.submenu = appMenu
    NSApp.mainMenu = mainMenu
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
installMenu()
let delegate = AppDelegate()
app.delegate = delegate
app.run()
