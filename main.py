import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == "darwin":  
    os.environ["QT_QPA_PLATFORM"] = "cocoa"
    os.environ["QT_MAC_WANTS_LAYER"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    
    if not os.environ.get("CFBundleName"):
        os.environ["CFBundleName"] = "Fragmenta Desktop"
    if not os.environ.get("CFBundleDisplayName"):
        os.environ["CFBundleDisplayName"] = "Fragmenta Desktop"
    if not os.environ.get("CFBundleIdentifier"):
        os.environ["CFBundleIdentifier"] = "com.misaghazimi.fragmenta"
    
    print(f"macOS detected, Qt platform set to: {os.environ.get('QT_QPA_PLATFORM', 'not set')}")
    print(f" App bundle name: {os.environ.get('CFBundleDisplayName', 'not set')}")

    try:
        import setproctitle
        setproctitle.setproctitle("Fragmenta Desktop")
        print("Process name set via setproctitle")
    except ImportError:
        print("setproctitle not available - process may show as 'Python'")
        sys.argv[0] = "Fragmenta Desktop"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from app.desktop.main import main
    main()
