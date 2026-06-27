#!/usr/bin/env bash
#
# build.sh — compile EMFieldDemo (Swift + Metal, macOS).
#
#   ./build.sh          build the optimized binary       -> ./EMFieldDemo
#   ./build.sh run      build, then run
#   ./build.sh app      build, then wrap into a bundle   -> ./EMFieldDemo.app
#   ./build.sh clean    remove build artifacts
#
set -euo pipefail

SRC="EMFieldDemo.swift"
BIN="EMFieldDemo"
APP="EMFieldDemo.app"
BUNDLE_ID="com.nightmarez.emfielddemo"

die() { echo "error: $*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "this demo requires macOS (Metal)."
command -v swiftc >/dev/null 2>&1 \
    || die "swiftc not found — install Xcode or the Command Line Tools (xcode-select --install)."
[[ -f "$SRC" ]] || die "$SRC not found (run this from the project directory)."

build() {
    echo "==> compiling $SRC (release)"
    swiftc -O "$SRC" -o "$BIN" \
        -framework Cocoa -framework Metal -framework MetalKit
    echo "==> built ./$BIN"
}

bundle() {
    build
    echo "==> packaging $APP"
    rm -rf "$APP"
    mkdir -p "$APP/Contents/MacOS"
    cp "$BIN" "$APP/Contents/MacOS/$BIN"
    cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>               <string>EMFieldDemo</string>
    <key>CFBundleDisplayName</key>        <string>EM Field Demo</string>
    <key>CFBundleExecutable</key>         <string>$BIN</string>
    <key>CFBundleIdentifier</key>         <string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key>            <string>1.0</string>
    <key>CFBundleShortVersionString</key> <string>1.0</string>
    <key>CFBundlePackageType</key>        <string>APPL</string>
    <key>NSPrincipalClass</key>           <string>NSApplication</string>
    <key>NSHighResolutionCapable</key>    <true/>
    <key>LSMinimumSystemVersion</key>     <string>11.0</string>
</dict>
</plist>
PLIST
    echo "==> built ./$APP"
    echo "    note: PNG capture (P) writes to the working directory; when launched"
    echo "    by double-clicking, that is '/'. Run ./$BIN from a terminal to save frames."
}

case "${1:-build}" in
    build) build ;;
    run)   build; echo "==> running"; "./$BIN" ;;
    app)   bundle ;;
    clean) rm -rf "$BIN" "$APP"; echo "==> cleaned" ;;
    *)     die "unknown command: ${1:-} (use: build | run | app | clean)" ;;
esac
