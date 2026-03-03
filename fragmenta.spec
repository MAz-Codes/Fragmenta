# fragmenta.spec
# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# 1. Collect Valid Data Files
# -------------------------
datas = [
    ('app/frontend/build', 'app/frontend/build'),  # React App
    ('config/*.json', 'config'),                   # Root Configs (e.g. terms_accepted.json)
    ('models/config', 'models/config'),            # Model Configs
    ('README.md', '.'),
    ('LICENSE', '.'),
    ('NOTICE.md', '.'),
]

# 2. Collect Hidden Imports
# -----------------------
# Removed unused sklearn and flask_socketio.
# Kept core scientific and torch libraries.
hiddenimports = [
    'engineio.async_drivers.threading', 
    'torch',
    'torchaudio',
    'soundfile',
    # 'scipy.special.cython_special', # Usually covered by collect_all below, but keeping if explicit needed
]

# Collect all hooks and sanitize outputs
binaries = []
def collect_and_add(package_name):
    global datas, hiddenimports, binaries
    try:
        tmp_ret = collect_all(package_name)
        
        # 1. Datas: Expect list of (source, dest)
        datas += tmp_ret[0]
        
        # 2. Hidden Imports: Expect list of strings.
        #    If tuples found (like in torchaudio hook), move to binaries.
        for item in tmp_ret[1]:
            if isinstance(item, str):
                hiddenimports.append(item)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                binaries.append(item)
                
        # 3. Binaries: Expect list of (source, dest).
        #    Some hooks return random strings in binaries list (e.g. package name), filter them out.
        for item in tmp_ret[2]:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                binaries.append(item)
                
    except Exception as e:
        print(f"WARNING: Error collecting {package_name}: {e}")

collect_and_add('scipy')
collect_and_add('torchaudio')
collect_and_add('librosa')

# 3. Main Analysis
# ----------------
a = Analysis(
    ['app/desktop/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['ipython', 'jupyter', 'tkinter', 'notebook', 'matplotlib.tests', 'numpy.tests'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# 4. Create the Collect Bundle (Folder)
# -----------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FragmentaDesktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='scripts/icon.ico' if sys.platform == 'win32' else 'scripts/fragmenta_icon_1024.png'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FragmentaDesktop',
)

# 5. MacOS App Bundle (Optional)
# ----------------------------
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='FragmentaDesktop.app',
        icon='scripts/fragmenta_icon_1024.png',
        bundle_identifier='com.misaghazimi.fragmenta',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSBackgroundOnly': 'False',
        },
    )
