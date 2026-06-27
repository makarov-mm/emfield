// =============================================================================
//  EMFieldDemo.swift
//  Charged particle in an electromagnetic field + field-line tracing.
//
//  Scenarios:
//      1  Magnetic dipole      (radiation-belt-like trapping)
//      2  Magnetic quadrupole  (focusing / hyperbolic field lines)
//      3  Cyclotron            (uniform B -> helical gyration)
//      4  Magnetic bottle      (magnetic mirror -> bouncing motion)
//      5  Electric dipole      (Coulomb E-field lines, accelerated test charge)
//      6  Electric quadrupole  (4 point charges, deflection trajectory)
//
//  Stack: AppKit + MetalKit + simd + ImageIO. No third-party dependencies.
//  The Metal shader source is compiled at runtime (makeLibrary(source:)),
//  so there is no .metallib build step and no Xcode project required.
//
//  Build:
//      swiftc EMFieldDemo.swift -o EMFieldDemo \
//             -framework Cocoa -framework Metal -framework MetalKit
//  Run:
//      ./EMFieldDemo
//
//  Controls:
//      1..6      switch scenario
//      Space     pause / resume
//      R         reset particle
//      F         toggle field lines
//      T         toggle trajectory trail
//      A         toggle camera auto-rotation
//      P         save a 1920x1080 PNG of the current frame (to CWD)
//      drag      orbit camera
//      scroll    zoom
//
//  Magnetic scenarios: Boris pusher (energy-preserving for the magnetic part).
//  Electric scenarios: same integrator, the E half-kicks now do real work, so
//  the test charge gains/loses energy and is re-injected when it leaves view.
//  Field lines are traced on the CPU with RK4 along the normalized field and
//  coloured by local magnitude (log-scaled).
// =============================================================================

import Cocoa
import Metal
import MetalKit
import simd
import ImageIO
import UniformTypeIdentifiers

// MARK: - Tunables

private let kMaxTrail = 2400         // trajectory trail capacity (vertices)
private let kReinject = Float(5.5)   // re-inject the particle when |pos| exceeds this

// MARK: - Vertex / uniform layout (must match the MSL structs below)

struct LineVertex {
    var position: SIMD3<Float>
    var color:    SIMD4<Float>
}

struct Uniforms {
    var viewProj:    float4x4         // 64 bytes
    var particlePos: SIMD3<Float>     // 16 (aligned)
    var cameraRight: SIMD3<Float>     // 16
    var cameraUp:    SIMD3<Float>     // 16
    var pointSize:   Float            // 4
    var _pad0:       Float = 0        // padding -> stride stays 16-aligned (128)
    var _pad1:       SIMD2<Float> = .zero
}

@inline(__always) func rgba(_ c: SIMD3<Float>, _ a: Float) -> SIMD4<Float> {
    SIMD4<Float>(c.x, c.y, c.z, a)
}

// MARK: - Math helpers (Metal-compatible matrices, NDC z in [0,1])

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
    let z = normalize(eye - center)        // camera looks down -z
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

/// Cool-to-warm 5-stop colour map, input clamped to [0,1].
func colormap(_ t: Float) -> SIMD3<Float> {
    let stops: [SIMD3<Float>] = [
        SIMD3(0.10, 0.20, 0.78),   // deep blue
        SIMD3(0.10, 0.74, 0.86),   // cyan
        SIMD3(0.30, 0.85, 0.38),   // green
        SIMD3(0.96, 0.85, 0.26),   // gold
        SIMD3(0.96, 0.30, 0.20)    // red
    ]
    let x = max(0, min(1, t)) * 4.0
    let i = min(3, Int(x))
    return mix(stops[i], stops[i + 1], t: x - Float(i))
}

// MARK: - Scenario definition

enum LineField { case magnetic, electric }

struct Scenario {
    let name: String
    /// Returns the (E, B) field at a point.
    let field: (SIMD3<Float>) -> (E: SIMD3<Float>, B: SIMD3<Float>)
    let q: Float
    let m: Float
    let dt: Float
    let substeps: Int
    let initialPos: SIMD3<Float>
    let initialVel: SIMD3<Float>
    let cameraDistance: Float
    let seeds: [SIMD3<Float>]            // field-line seed points
    let uniformLineColor: SIMD3<Float>?  // non-nil for spatially uniform fields
    // defaults below keep the existing memberwise calls valid
    var lineField: LineField = .magnetic // which field to trace lines along
    var stopSites: [SIMD3<Float>] = []   // point charges: stop tracing nearby
}

enum Scenarios {

    static let count = 6

    // --- 1. Magnetic dipole (axis = z) -------------------------------------
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
            initialPos: SIMD3(1.7, 0.0, 0.25),
            initialVel: SIMD3(0.0, 1.5, 0.45),
            cameraDistance: 7.0, seeds: seeds, uniformLineColor: nil)
    }

    // --- 2. Magnetic quadrupole (B in the xy-plane) ------------------------
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
            initialPos: SIMD3(0.65, 0.05, -2.6),
            initialVel: SIMD3(0.0, 0.0, 1.6),
            cameraDistance: 7.5, seeds: seeds, uniformLineColor: nil)
    }

    // --- 3. Cyclotron (uniform B along z) ----------------------------------
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
            initialPos: SIMD3(1.0, 0.0, -1.6),
            initialVel: SIMD3(0.0, 1.2, 0.55),
            cameraDistance: 7.0, seeds: seeds,
            uniformLineColor: SIMD3(0.25, 0.55, 0.95))
    }

    // --- 4. Magnetic bottle / mirror (axis = z) ----------------------------
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
            initialPos: SIMD3(0.5, 0.0, 0.0),
            initialVel: SIMD3(0.0, 1.0, 0.8),
            cameraDistance: 7.0, seeds: seeds, uniformLineColor: nil)
    }

    // --- Coulomb field of a set of point charges (k = 1) -------------------
    static func coulomb(_ p: SIMD3<Float>, _ charges: [(SIMD3<Float>, Float)]) -> SIMD3<Float> {
        var e = SIMD3<Float>(0, 0, 0)
        for (c, qc) in charges {
            let d = p - c
            let r = max(length(d), 0.28)   // softening
            e += qc * d / (r * r * r)
        }
        return e
    }

    /// Seed a ring bundle around a (positive) charge, from which E-lines emanate.
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

    // --- 5. Electric dipole (+q and -q on the x-axis) ----------------------
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
            initialPos: SIMD3(0.0, 1.9, 0.25),
            initialVel: SIMD3(0.85, 0.0, 0.0),
            cameraDistance: 7.5, seeds: seeds, uniformLineColor: nil,
            lineField: .electric, stopSites: charges.map { $0.0 })
    }

    // --- 6. Electric quadrupole (+,+ on x-axis, -,- on y-axis) -------------
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
            initialPos: SIMD3(2.3, 0.35, 0.0),
            initialVel: SIMD3(-1.05, 0.15, 0.0),
            cameraDistance: 7.8, seeds: seeds, uniformLineColor: nil,
            lineField: .electric, stopSites: charges.map { $0.0 })
    }

    static func make(_ index: Int) -> Scenario {
        switch index {
        case 0:  return dipole()
        case 1:  return quadrupole()
        case 2:  return cyclotron()
        case 3:  return bottle()
        case 4:  return electricDipole()
        default: return electricQuadrupole()
        }
    }
}

// MARK: - Physics: Boris pusher

func borisStep(pos: inout SIMD3<Float>, vel: inout SIMD3<Float>,
               q: Float, m: Float, dt: Float,
               field: (SIMD3<Float>) -> (E: SIMD3<Float>, B: SIMD3<Float>)) {
    let f = field(pos)
    let h = (q / m) * dt * 0.5
    let vMinus = vel + h * f.E                 // half E kick
    let t  = h * f.B                            // rotation by B
    let s  = (2.0 / (1.0 + dot(t, t))) * t
    let vP = vMinus + cross(vMinus, t)
    let vPlus = vMinus + cross(vP, s)
    vel = vPlus + h * f.E                       // second half E kick
    pos = pos + vel * dt
}

// MARK: - Field-line tracing (CPU, RK4 along normalized field)

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
            if hit { pts.append(p); break }     // terminate the line on the sink charge
        }
        pts.append(p)
    }
    return pts
}

/// Builds line-segment geometry (MTLPrimitiveType.line) for all seeds,
/// coloured by log magnitude of the traced field.
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

    func colorAt(_ p: SIMD3<Float>) -> SIMD4<Float> {
        if let c = s.uniformLineColor { return rgba(c, 0.85) }
        let t = (log(length(fvec(p)) + 1e-6) - lo) / span
        return rgba(colormap(t), 0.85)
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

/// Static reference axes (dim), drawn as a line list.
func buildAxes() -> [LineVertex] {
    let L: Float = 3.0
    let a: Float = 0.22
    func seg(_ p0: SIMD3<Float>, _ p1: SIMD3<Float>, _ c: SIMD3<Float>) -> [LineVertex] {
        [LineVertex(position: p0, color: rgba(c, a)),
         LineVertex(position: p1, color: rgba(c, a))]
    }
    return seg(SIMD3(-L,0,0), SIMD3(L,0,0), SIMD3(0.9,0.3,0.3))
         + seg(SIMD3(0,-L,0), SIMD3(0,L,0), SIMD3(0.3,0.9,0.3))
         + seg(SIMD3(0,0,-L), SIMD3(0,0,L), SIMD3(0.3,0.5,0.95))
}

// MARK: - World (shared mutable state, main thread only)

final class World {
    private(set) var scenarioIndex = 0
    private(set) var scenario = Scenarios.make(0)

    var pos = SIMD3<Float>(0, 0, 0)
    var vel = SIMD3<Float>(0, 0, 0)
    var trail: [SIMD3<Float>] = []

    var paused = false
    var showFieldLines = true
    var showTrail = true
    var autoRotate = true

    var azimuth: Float = 0.7
    var elevation: Float = 0.4
    var distance: Float = 7.0

    private(set) var fieldLineVerts: [LineVertex] = []
    private(set) var axisVerts: [LineVertex] = buildAxes()
    var fieldLinesDirty = true

    init() { loadScenario(0) }

    func loadScenario(_ i: Int) {
        scenarioIndex = ((i % Scenarios.count) + Scenarios.count) % Scenarios.count
        scenario = Scenarios.make(scenarioIndex)
        distance = scenario.cameraDistance
        resetParticle()
        fieldLineVerts = buildFieldLines(scenario)
        fieldLinesDirty = true
    }

    func resetParticle() {
        pos = scenario.initialPos
        vel = scenario.initialVel
        trail.removeAll(keepingCapacity: true)
    }

    func step() {
        guard !paused else { return }
        let s = scenario
        for _ in 0..<s.substeps {
            borisStep(pos: &pos, vel: &vel, q: s.q, m: s.m, dt: s.dt, field: s.field)
            if length(pos) > kReinject { resetParticle(); break }
            trail.append(pos)
            if trail.count > kMaxTrail { trail.removeFirst(trail.count - kMaxTrail) }
        }
    }

    func trailVerts() -> [LineVertex] {
        guard trail.count > 1 else { return [] }
        let n = trail.count
        var out: [LineVertex] = []
        out.reserveCapacity(n)
        for (i, p) in trail.enumerated() {
            let age = Float(i) / Float(n - 1)
            let c = mix(SIMD3<Float>(0.35, 0.55, 1.0),
                        SIMD3<Float>(0.6, 1.0, 1.0), t: age)
            out.append(LineVertex(position: p, color: rgba(c, 0.15 + 0.85 * age)))
        }
        return out
    }
}

// MARK: - Metal shader source (compiled at runtime)

private let kShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct LineVertex { float3 position; float4 color; };

struct Uniforms {
    float4x4 viewProj;
    float3   particlePos;
    float3   cameraRight;
    float3   cameraUp;
    float    pointSize;
};

struct LineOut { float4 position [[position]]; float4 color; };

vertex LineOut line_vertex(const device LineVertex* verts [[buffer(0)]],
                           constant Uniforms& u           [[buffer(1)]],
                           uint vid                       [[vertex_id]]) {
    LineOut o;
    o.position = u.viewProj * float4(verts[vid].position, 1.0);
    o.color    = verts[vid].color;
    return o;
}

fragment float4 line_fragment(LineOut in [[stage_in]]) {
    return in.color;
}

struct GlowOut { float4 position [[position]]; float2 uv; float4 color; };

vertex GlowOut glow_vertex(constant Uniforms& u  [[buffer(1)]],
                           constant float4& glow [[buffer(2)]],
                           uint vid              [[vertex_id]]) {
    float2 corners[4] = { float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1) };
    float2 c = corners[vid];
    float3 wp = u.particlePos + (c.x * u.cameraRight + c.y * u.cameraUp) * u.pointSize;
    GlowOut o;
    o.position = u.viewProj * float4(wp, 1.0);
    o.uv = c;
    o.color = glow;
    return o;
}

fragment float4 glow_fragment(GlowOut in [[stage_in]]) {
    float r = length(in.uv);
    float a = exp(-r * r * 3.0);                 // soft gaussian falloff
    a = clamp(a, 0.0, 1.0);
    return float4(in.color.rgb * a * in.color.a, a * in.color.a);  // premultiplied
}
"""

// MARK: - Renderer

final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let world: World

    private var linePipeline: MTLRenderPipelineState!
    private var glowPipeline: MTLRenderPipelineState!
    private var dsLines: MTLDepthStencilState!
    private var dsGlow:  MTLDepthStencilState!

    private var fieldBuffer: MTLBuffer?
    private var axisBuffer:  MTLBuffer?
    private var trailBuffer: MTLBuffer!

    private var aspect: Float = 1.0
    private let clear = MTLClearColor(red: 0.02, green: 0.02, blue: 0.05, alpha: 1.0)

    init?(view: MTKView, world: World) {
        guard let dev = view.device ?? MTLCreateSystemDefaultDevice(),
              let q = dev.makeCommandQueue() else { return nil }
        self.device = dev; self.queue = q; self.world = world
        super.init()

        view.colorPixelFormat = .bgra8Unorm
        view.depthStencilPixelFormat = .depth32Float
        view.clearColor = clear

        do { try buildPipelines(view: view) }
        catch { print("Pipeline build failed: \(error)"); return nil }

        trailBuffer = device.makeBuffer(length: MemoryLayout<LineVertex>.stride * kMaxTrail,
                                        options: .storageModeShared)
        axisBuffer = makeLineBuffer(world.axisVerts)
        refreshFieldBuffer()
    }

    private func buildPipelines(view: MTKView) throws {
        let lib = try device.makeLibrary(source: kShaderSource, options: nil)

        func pipeline(_ vfn: String, _ ffn: String, additive: Bool) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction   = lib.makeFunction(name: vfn)
            d.fragmentFunction = lib.makeFunction(name: ffn)
            d.depthAttachmentPixelFormat = view.depthStencilPixelFormat
            let ca = d.colorAttachments[0]!
            ca.pixelFormat = view.colorPixelFormat
            ca.isBlendingEnabled = true
            ca.rgbBlendOperation = .add
            ca.alphaBlendOperation = .add
            if additive {
                ca.sourceRGBBlendFactor = .one
                ca.sourceAlphaBlendFactor = .one
                ca.destinationRGBBlendFactor = .one
                ca.destinationAlphaBlendFactor = .one
            } else {
                ca.sourceRGBBlendFactor = .sourceAlpha
                ca.sourceAlphaBlendFactor = .one
                ca.destinationRGBBlendFactor = .oneMinusSourceAlpha
                ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }
            return try device.makeRenderPipelineState(descriptor: d)
        }

        linePipeline = try pipeline("line_vertex", "line_fragment", additive: false)
        glowPipeline = try pipeline("glow_vertex", "glow_fragment", additive: true)

        let dl = MTLDepthStencilDescriptor()
        dl.depthCompareFunction = .less; dl.isDepthWriteEnabled = true
        dsLines = device.makeDepthStencilState(descriptor: dl)

        let dg = MTLDepthStencilDescriptor()
        dg.depthCompareFunction = .always; dg.isDepthWriteEnabled = false
        dsGlow = device.makeDepthStencilState(descriptor: dg)
    }

    private func makeLineBuffer(_ verts: [LineVertex]) -> MTLBuffer? {
        guard !verts.isEmpty else { return nil }
        return device.makeBuffer(bytes: verts,
                                 length: MemoryLayout<LineVertex>.stride * verts.count,
                                 options: .storageModeShared)
    }

    func refreshFieldBuffer() {
        fieldBuffer = makeLineBuffer(world.fieldLineVerts)
        world.fieldLinesDirty = false
    }

    // Camera + uniforms for a given aspect ratio.
    private func makeUniforms(aspect: Float) -> Uniforms {
        let el = max(-1.45, min(1.45, world.elevation))
        let az = world.azimuth
        let eye = SIMD3<Float>(world.distance * cos(el) * cos(az),
                               world.distance * sin(el),
                               world.distance * cos(el) * sin(az))
        let target = SIMD3<Float>(0, 0, 0)
        let worldUp = SIMD3<Float>(0, 1, 0)
        let forward = normalize(target - eye)
        let right = normalize(cross(forward, worldUp))
        let camUp = cross(right, forward)
        let viewM = lookAtRH(eye: eye, center: target, up: worldUp)
        let projM = perspectiveRH(fovyRadians: 0.9, aspect: aspect, near: 0.05, far: 100)
        return Uniforms(viewProj: projM * viewM, particlePos: world.pos,
                        cameraRight: right, cameraUp: camUp, pointSize: 0.18)
    }

    // Shared encode path for both the live view and the offscreen PNG capture.
    private func encodeScene(_ enc: MTLRenderCommandEncoder, _ base: Uniforms) {
        var u = base
        enc.setRenderPipelineState(linePipeline)
        enc.setDepthStencilState(dsLines)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)

        if let ab = axisBuffer {
            enc.setVertexBuffer(ab, offset: 0, index: 0)
            enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: world.axisVerts.count)
        }
        if world.showFieldLines, let fb = fieldBuffer {
            enc.setVertexBuffer(fb, offset: 0, index: 0)
            enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: world.fieldLineVerts.count)
        }
        if world.showTrail {
            let tv = world.trailVerts()
            if tv.count > 1 {
                tv.withUnsafeBytes { raw in
                    trailBuffer.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
                }
                enc.setVertexBuffer(trailBuffer, offset: 0, index: 0)
                enc.drawPrimitives(type: .lineStrip, vertexStart: 0, vertexCount: tv.count)
            }
        }

        // particle glow: additive halo + bright core
        enc.setRenderPipelineState(glowPipeline)
        enc.setDepthStencilState(dsGlow)

        var halo = base; halo.pointSize = 0.42
        enc.setVertexBytes(&halo, length: MemoryLayout<Uniforms>.stride, index: 1)
        var haloColor = SIMD4<Float>(0.3, 0.7, 1.0, 0.5)
        enc.setVertexBytes(&haloColor, length: MemoryLayout<SIMD4<Float>>.stride, index: 2)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)

        var core = base; core.pointSize = 0.13
        enc.setVertexBytes(&core, length: MemoryLayout<Uniforms>.stride, index: 1)
        var coreColor = SIMD4<Float>(1.0, 1.0, 1.0, 1.0)
        enc.setVertexBytes(&coreColor, length: MemoryLayout<SIMD4<Float>>.stride, index: 2)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    // MARK: MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        aspect = Float(max(size.width, 1) / max(size.height, 1))
    }

    func draw(in view: MTKView) {
        if world.fieldLinesDirty { refreshFieldBuffer() }
        world.step()

        guard let rpd = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }

        encodeScene(enc, makeUniforms(aspect: aspect))
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()

        if world.autoRotate { world.azimuth += 0.0025 }
    }

    // MARK: PNG capture (offscreen render -> blit to buffer -> ImageIO)

    func capture(width W: Int = 1920, height H: Int = 1080) {
        let cd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                          width: W, height: H, mipmapped: false)
        cd.usage = [.renderTarget, .shaderRead]; cd.storageMode = .private
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float,
                                                          width: W, height: H, mipmapped: false)
        dd.usage = [.renderTarget]; dd.storageMode = .private
        guard let colorTex = device.makeTexture(descriptor: cd),
              let depthTex = device.makeTexture(descriptor: dd) else { return }

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = colorTex
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].clearColor = clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.depthAttachment.texture = depthTex
        rpd.depthAttachment.loadAction = .clear
        rpd.depthAttachment.clearDepth = 1.0
        rpd.depthAttachment.storeAction = .dontCare

        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: rpd) else { return }
        encodeScene(enc, makeUniforms(aspect: Float(W) / Float(H)))
        enc.endEncoding()

        let rowBytes = ((W * 4 + 255) / 256) * 256          // 256-byte aligned for blit
        guard let outBuf = device.makeBuffer(length: rowBytes * H, options: .storageModeShared),
              let blit = cmd.makeBlitCommandEncoder() else { return }
        blit.copy(from: colorTex, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: W, height: H, depth: 1),
                  to: outBuf, destinationOffset: 0,
                  destinationBytesPerRow: rowBytes,
                  destinationBytesPerImage: rowBytes * H)
        blit.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        writePNG(outBuf.contents(), width: W, height: H, rowBytes: rowBytes)
    }

    private func writePNG(_ data: UnsafeMutableRawPointer, width: Int, height: Int, rowBytes: Int) {
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { return }
        // BGRA bytes read as a little-endian 32-bit word => skip the high (alpha) byte.
        let bmp = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(data: data, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: rowBytes,
                                  space: cs, bitmapInfo: bmp),
              let img = ctx.makeImage() else { return }

        let name = "emfield_\(Int(Date().timeIntervalSince1970)).png"
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(name)
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL,
                                                         UTType.png.identifier as CFString,
                                                         1, nil) else { return }
        CGImageDestinationAddImage(dest, img, nil)
        if CGImageDestinationFinalize(dest) { print("Saved \(url.path)") }
        else { print("PNG export failed") }
    }
}

// MARK: - View (input handling)

final class DemoView: MTKView {
    weak var world: World?
    var onScenarioChange: (() -> Void)?
    var onCapture: (() -> Void)?

    override var acceptsFirstResponder: Bool { true }

    override func keyDown(with event: NSEvent) {
        guard let w = world else { return }
        switch event.charactersIgnoringModifiers ?? "" {
        case "1": switchTo(0)
        case "2": switchTo(1)
        case "3": switchTo(2)
        case "4": switchTo(3)
        case "5": switchTo(4)
        case "6": switchTo(5)
        case " ": w.paused.toggle()
        case "r", "R": w.resetParticle()
        case "f", "F": w.showFieldLines.toggle()
        case "t", "T": w.showTrail.toggle()
        case "a", "A": w.autoRotate.toggle()
        case "p", "P": onCapture?()
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
        window?.title = "EM Field Demo  —  [\(world?.scenario.name ?? "")]   "
            + "(1-6 switch · Space pause · R reset · F lines · T trail · A spin · P png)"
    }
}

// MARK: - App bootstrap

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var renderer: Renderer!
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
        view.onScenarioChange = { [weak r] in r?.refreshFieldBuffer() }
        view.onCapture = { [weak r] in r?.capture() }

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
