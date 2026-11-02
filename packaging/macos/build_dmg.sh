#!/usr/bin/env bash
# Build Fragmenta.app and its .dmg (Apple Silicon). Runs on a Mac; CI runs it too.
#
#   Fragmenta.app/Contents/
#   ├── MacOS/fragmenta            frozen packaging/launcher.py
#   ├── Frameworks/                PyInstaller's Python + Tk for the launcher
#   ├── Info.plist  Resources/fragmenta.icns
#   └── Resources/fragmenta/       the bundle: runtime/{python,uv} + payload/
#
# No Developer ID and no notarization (D22). Apple Silicon only runs signed code,
# so everything gets an ad-hoc signature instead. A downloaded copy then opens
# after System Settings → Privacy & Security → Open Anyway. Without the hardened
# runtime, the unsigned packages installed at first run load without entitlements.
#
# Env: BUILD_NUMBER  CFBundleVersion, rising with every build (CI: its run number).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
VERSION="$(tr -d '[:space:]' < VERSION)"
SHORT_VERSION="${VERSION%%-*}"            # 2.0.0-rc.1 -> 2.0.0: macOS wants numbers
BUILD_NUMBER="${BUILD_NUMBER:-$SHORT_VERSION}"
DIST="packaging/build"
BUNDLE="$DIST/bundle"
STAGE="$DIST/dmg"                         # what the .dmg window shows
APP="$STAGE/Fragmenta.app"
DMG="$DIST/Fragmenta-${VERSION}-macOS-arm64.dmg"
PYINSTALLER="pyinstaller==6.22.3"

echo "==> Fragmenta $VERSION (build $BUILD_NUMBER), macOS arm64"

# 1. Bundle: runtime/python + runtime/uv + payload/.
python3 packaging/assemble.py --target macos-arm64 --out "$REPO/$BUNDLE" --ref "${FRAGMENTA_REF:-HEAD}"

# 2. Freeze the launcher with the Python we ship: it carries Tk for the first-run
# window, so the build Mac needs none. PyInstaller sits in a throwaway venv, so it
# never lands in the shipped Python.
PBS_PY="$BUNDLE/runtime/python/bin/python3"
BUILD_VENV="$DIST/_buildvenv"
"$PBS_PY" -c "import tkinter" 2>/dev/null || {
    echo "ERROR: the bundled Python has no Tkinter, so the launcher window would be missing." >&2
    exit 1
}
rm -rf "$BUILD_VENV" "$STAGE" "$DIST/_pyi"
"$PBS_PY" -m venv "$BUILD_VENV"
"$BUILD_VENV/bin/python" -m pip install --quiet "$PYINSTALLER"
mkdir -p "$STAGE"
"$BUILD_VENV/bin/python" -m PyInstaller --noconfirm --windowed --name fragmenta \
    --osx-bundle-identifier com.misaghazimi.fragmenta --target-arch arm64 \
    --distpath "$DIST/_pyi/dist" --workpath "$DIST/_pyi/work" --specpath "$DIST/_pyi" \
    packaging/launcher.py
mv "$DIST/_pyi/dist/fragmenta.app" "$APP"

# 3. Bundle into its own folder under Resources; Info.plist + icon.
ditto "$BUNDLE" "$APP/Contents/Resources/fragmenta"
sed -e "s/@@VERSION@@/$SHORT_VERSION/g" -e "s/@@BUILD@@/$BUILD_NUMBER/g" \
    packaging/macos/Info.plist.in > "$APP/Contents/Info.plist"
plutil -lint "$APP/Contents/Info.plist" >/dev/null
ICON_PNG="app/frontend/public/fragmenta_icon_1024.png"
ICONSET="$DIST/fragmenta.iconset"
rm -rf "$ICONSET"; mkdir -p "$ICONSET"
for sz in 16 32 64 128 256 512; do
    sips -z $sz $sz   "$ICON_PNG" --out "$ICONSET/icon_${sz}x${sz}.png"   >/dev/null
    sips -z $((sz*2)) $((sz*2)) "$ICON_PNG" --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/fragmenta.icns"

# 4. Ad-hoc signature: every Mach-O file first, then the app, which seals the
# rest. Any failure stops the build.
echo "==> ad-hoc signature"
find "$APP/Contents" -type f ! -path "$APP/Contents/MacOS/fragmenta" -print0 \
    | xargs -0 file -N -F $'\t' | awk -F'\t' '$2 ~ /Mach-O/ {print $1}' > "$DIST/_macho.txt"
echo "    $(wc -l < "$DIST/_macho.txt" | tr -d ' ') Mach-O files"
tr '\n' '\0' < "$DIST/_macho.txt" | xargs -0 codesign --force --sign -
codesign --force --sign - "$APP"
codesign --verify --strict --deep "$APP"

# 5. .dmg: the app beside a link to Applications.
echo "==> dmg"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
for attempt in 1 2 3; do  # hdiutil sometimes reports a busy device on CI
    hdiutil create -volname "Fragmenta $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG" && break
    [ "$attempt" = 3 ] && exit 1
    sleep 5
done
shasum -a 256 "$DMG"
echo "==> done: $DMG"
