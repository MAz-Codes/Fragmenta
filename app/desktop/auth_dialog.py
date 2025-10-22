import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    try:
        from PyQt6.QtWidgets import QApplication
        from app.core.hf_auth_dialog import show_hf_auth_dialog

        print("Fragmenta Desktop - Authentication Dialog")
        print("=" * 50)
        print("Opening Hugging Face authentication dialog...")
        print("This dialog will guide you through:")
        print("1. Accepting model terms")
        print("2. Getting your access token")
        print("3. Logging in with your token")
        print("4. Testing authentication")
        print("5. Downloading selected models")
        print()

        app = QApplication(sys.argv)
        app.setApplicationName("Fragmenta Desktop - Authentication")

        result = show_hf_auth_dialog()

        if result:
            print("Authentication dialog completed successfully")
        else:
            print("Authentication dialog was cancelled or closed")

    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure you're running this from the Fragmenta Desktop directory")
        print("and the virtual environment is activated:")
        print("source venv/bin/activate")
        print("python auth_dialog.py")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
