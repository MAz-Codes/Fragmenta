#!/usr/bin/env bash
# Build the Linux AppImage and a tarball of the same tree.
#
#   Fragmenta.AppDir/
#   ├── AppRun  fragmenta.desktop  fragmenta.png
#   ├── runtime/{python,uv}
#   └── payload/    install.py  start.py  launcher.py  app/ ...
#
# Nothing here compiles: the Python, uv and appimagetool are prebuilt and pinned,
# so any x86_64 Linux host can run this. Run from anywhere.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/packaging/build"
APPDIR="$BUILD/Fragmenta.AppDir"
TOOLS="$BUILD/tools"
VERSION="$(cat "$ROOT/VERSION")"
NAME="Fragmenta-$VERSION-Linux-x86_64"

fetch() {  # url sha256 dest
    [ -f "$3" ] && echo "$2  $3" | sha256sum -c --quiet - 2>/dev/null && return 0
    curl -fsSL --retry 3 -o "$3" "$1"
    echo "$2  $3" | sha256sum -c --quiet -
}

mkdir -p "$TOOLS"
fetch https://github.com/AppImage/appimagetool/releases/download/1.9.1/appimagetool-x86_64.AppImage \
    ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0 "$TOOLS/appimagetool"
fetch https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64 \
    2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d "$TOOLS/runtime-x86_64"
chmod +x "$TOOLS/appimagetool"

python3 "$ROOT/packaging/assemble.py" --target linux-x86_64 --out "$APPDIR" --ref "${FRAGMENTA_REF:-HEAD}"

# assemble.py leaves packaging/ out of the payload, and Linux has no frozen launcher.
cp "$ROOT/packaging/launcher.py" "$APPDIR/payload/launcher.py"
cp "$ROOT/app/frontend/public/fragmenta_icon_1024.png" "$APPDIR/fragmenta.png"
ln -sf fragmenta.png "$APPDIR/.DirIcon"

cat > "$APPDIR/fragmenta.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Fragmenta
Comment=Fine-tune and use text-to-audio models
Exec=fragmenta
Icon=fragmenta
Categories=AudioVideo;Audio;
DESKTOP

cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export FRAGMENTA_BUNDLE="$HERE"
exec "$HERE/runtime/python/bin/python3" -I "$HERE/payload/launcher.py" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

echo "==> tarball"
tar -C "$BUILD" --transform "s|^Fragmenta.AppDir|$NAME|" -czf "$BUILD/$NAME.tar.gz" Fragmenta.AppDir

echo "==> AppImage"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$TOOLS/appimagetool" --no-appstream \
    --runtime-file "$TOOLS/runtime-x86_64" "$APPDIR" "$BUILD/$NAME.AppImage"

ls -la "$BUILD/$NAME.AppImage" "$BUILD/$NAME.tar.gz"
