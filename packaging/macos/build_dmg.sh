#!/usr/bin/env bash
# Build Fragmenta.app + .dmg (Apple Silicon).
#
# Runs on macOS (Apple Silicon). Steps:
#   1. PyInstaller freezes packaging/launcher.py -> Fragmenta.app skeleton.
#   2. Drop the assembled payload (app code + standalone Python) into Resources.
#   3. Install our Info.plist + .icns.
#   4. codesign (hardened runtime + entitlements) and notarize — if creds exist.
#   5. Wrap in a .dmg with create-dmg.
#
# Signing/notarization are SKIPPED when the relevant env vars are unset, so a
# plain local build still produces an (unsigned, Gatekeeper-blocked) .app/.dmg
# for smoke-testing.
#
# Env (optional, for a shippable signed build):
#   MAC_SIGN_IDENTITY   "Developer ID Application: Name (TEAMID)"
#   AC_APPLE_ID         Apple ID email           (notarization)
#   AC_TEAM_ID          Developer Team ID        (notarization)
#   AC_PASSWORD         app-specific password    (notarization)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
VERSION="$(tr -d '[:space:]' < VERSION)"
DIST="packaging/build"
APP="$DIST/Fragmenta.app"
PAYLOAD="$DIST/payload"
DMG="$DIST/Fragmenta-${VERSION}-macOS-arm64.dmg"

echo "==> Fragmenta $VERSION — macOS arm64 build"

# 1. Assemble the payload (app code + standalone Python).
python3 packaging/assemble.py --target macos-arm64 --out "$REPO/$PAYLOAD"

# 2. Freeze the launcher into an .app skeleton.
# The launcher's first-run progress splash uses Tkinter. Rather than depend on
# the build machine's system Python shipping Tk (it usually doesn't — hence the
# old 'brew install python-tk' requirement), we freeze with the SAME standalone
# CPython we ship: python-build-standalone bundles _tkinter + Tcl/Tk, so
# PyInstaller picks up the splash for free and the build needs nothing
# preinstalled. PyInstaller lives in a throwaway venv built from that interpreter
# so the shipped payload Python stays pristine (no PyInstaller riding along).
PBS_PY="$PAYLOAD/python-3.11/bin/python3.11"
BUILD_VENV="$DIST/_buildvenv"
"$PBS_PY" -c "import tkinter" 2>/dev/null || {
    echo "ERROR: the bundled standalone Python lacks Tkinter — the first-run splash" >&2
    echo "       would be missing. Bump PBS_RELEASE in packaging/python_standalone.py" >&2
    echo "       to a build that ships Tcl/Tk and rebuild." >&2
    exit 1
}
rm -rf "$BUILD_VENV"
"$PBS_PY" -m venv "$BUILD_VENV"
"$BUILD_VENV/bin/python" -m pip install --quiet --upgrade pip pyinstaller
rm -rf "$APP" "$DIST/fragmenta.app" build/launcher *.spec
"$BUILD_VENV/bin/python" -m PyInstaller --noconfirm --windowed --name fragmenta \
    --distpath "$DIST" --workpath "$DIST/_pyi" --specpath "$DIST/_pyi" \
    packaging/launcher.py
mv "$DIST/fragmenta.app" "$APP"

# 3. Payload -> Contents/Resources; Info.plist + icon.
cp -R "$PAYLOAD/." "$APP/Contents/Resources/"
sed "s/@@VERSION@@/$VERSION/g" packaging/macos/Info.plist.in > "$APP/Contents/Info.plist"
ICON_PNG="app/frontend/public/fragmenta_icon_1024.png"
ICONSET="$DIST/fragmenta.iconset"
rm -rf "$ICONSET"; mkdir -p "$ICONSET"
for sz in 16 32 64 128 256 512; do
    sips -z $sz $sz   "$ICON_PNG" --out "$ICONSET/icon_${sz}x${sz}.png"   >/dev/null
    sips -z $((sz*2)) $((sz*2)) "$ICON_PNG" --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/fragmenta.icns"

# 4. Sign + notarize (only if an identity is provided).
if [[ -n "${MAC_SIGN_IDENTITY:-}" ]]; then
    echo "==> codesign (hardened runtime)"
    # Sign nested Mach-O (bundled Python + any dylibs) first, then the app.
    find "$APP/Contents/Resources/python-3.11" -type f \( -name "*.dylib" -o -name "*.so" -o -perm -111 \) \
        -exec codesign --force --options runtime --timestamp \
        --entitlements packaging/macos/entitlements.plist --sign "$MAC_SIGN_IDENTITY" {} + || true
    codesign --force --options runtime --timestamp \
        --entitlements packaging/macos/entitlements.plist \
        --sign "$MAC_SIGN_IDENTITY" "$APP"

    if [[ -n "${AC_APPLE_ID:-}" && -n "${AC_TEAM_ID:-}" && -n "${AC_PASSWORD:-}" ]]; then
        echo "==> notarize"
        ditto -c -k --keepParent "$APP" "$DIST/_notarize.zip"
        xcrun notarytool submit "$DIST/_notarize.zip" \
            --apple-id "$AC_APPLE_ID" --team-id "$AC_TEAM_ID" --password "$AC_PASSWORD" --wait
        xcrun stapler staple "$APP"
    else
        echo "==> notarization creds absent — skipping (app will be Gatekeeper-blocked)"
    fi
else
    echo "==> MAC_SIGN_IDENTITY unset — unsigned build (local smoke-test only)"
fi

# 5. .dmg
echo "==> create-dmg"
rm -f "$DMG"
if command -v create-dmg >/dev/null 2>&1; then
    create-dmg --volname "Fragmenta $VERSION" \
        --app-drop-link 480 170 --icon "Fragmenta.app" 160 170 \
        --window-size 640 360 "$DMG" "$APP"
else
    echo "create-dmg not found; falling back to hdiutil"
    hdiutil create -volname "Fragmenta $VERSION" -srcfolder "$APP" -ov -format UDZO "$DMG"
fi
echo "==> done: $DMG"
