#!/usr/bin/env python3
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
elif sys.platform == "win32":
    os.environ["QT_QPA_PLATFORM"] = "windows"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
else:
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

from app.core.config import get_config
from app.core.model_manager import ModelManager
from app.backend.app import app as flask_app, QuietWSGIRequestHandler

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QUrl
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QMainWindow, QMenuBar, QMenu
    
    print(f"QT_QPA_PLATFORM environment: {os.environ.get('QT_QPA_PLATFORM', 'not set')}")
    
except ImportError as e:
    print(f"Failed to import PyQt6: {e}")
    sys.exit(1)

from utils.migration_helpers import enhanced_print
from utils.logger import setup_logging, get_logger
import threading
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

LOG_LEVEL = os.environ.get('FRAGMENTA_LOG_LEVEL', 'INFO')
setup_logging(log_level=LOG_LEVEL, log_file=True)


class FlaskThread(QThread):
    server_ready = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.logger = get_logger("FlaskThread")

    def run(self):
        try:
            time.sleep(0.5)

            self.logger.info("Starting Fragmenta...")
            self.logger.info(
                "Backend server starting on http://127.0.0.1:5001")

            self.server_ready.emit()

            flask_app.run(
                host='127.0.0.1',
                port=5001,
                debug=False,
                use_reloader=False,
                request_handler=QuietWSGIRequestHandler,
                threaded=True
            )
        except Exception as e:
            self.logger.error(f"Flask server error: {e}")


class FragmentaDesktop(QMainWindow):

    def __init__(self):
        super().__init__()

        self.logger = get_logger("FragmentaDesktop")
        self.logger.info("Initializing Fragmenta Desktop...")

        try:
            config = get_config()
            self.model_manager = ModelManager(config)

            self.init_ui()
            self.start_flask_server()

            QTimer.singleShot(2000, self.start_welcome_page_monitoring)

            self.logger.info("Fragmenta Desktop initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Fragmenta Desktop: {e}")
            raise

    def init_ui(self):

        self.setWindowTitle("Fragmenta Desktop")
        
        self.setWindowFilePath("fragmenta-desktop")
        
        icon_path = project_root / "scripts" / "fragmenta_icon_1024.png"
        if icon_path.exists():
            from PyQt6.QtGui import QIcon
            self.setWindowIcon(QIcon(str(icon_path)))

        screen = QApplication.primaryScreen().geometry()

        window_width = 1200
        window_height = 800

        x = (screen.width() - window_width) // 2
        y = (screen.height() - window_height) // 2

        self.setGeometry(x, y, window_width, window_height)

        min_width = 1000
        min_height = 700
        self.setMinimumSize(min_width, min_height)
        
        max_width = 2560
        max_height = 1600
        self.setMaximumSize(max_width, max_height)

        self.create_menu_bar()

        self.apply_menu_styling()

        from PyQt6.QtWidgets import QWidget, QVBoxLayout
        
        container = QWidget()
        layout = QVBoxLayout(container)
        
        margin_percentage = 0
        
        self.web_view = QWebEngineView()
        
        layout.setContentsMargins(
            int(window_width * margin_percentage / 100),
            int(window_height * margin_percentage / 100),
            int(window_width * margin_percentage / 100),
            int(window_height * margin_percentage / 100)
        )
        
        layout.addWidget(self.web_view)
        self.setCentralWidget(container)
        
        self.container = container
        self.layout = layout
        self.margin_percentage = margin_percentage

        self.web_view.setVisible(False)

        self.statusBar().hide()

    def create_menu_bar(self):

        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')

        auth_action = QAction('&Authentication & Model Setup', self)
        auth_action.setShortcut('Ctrl+Shift+A')
        auth_action.setStatusTip(
            'Open Hugging Face authentication and model download dialog')
        auth_action.triggered.connect(self.show_authentication_dialog)
        file_menu.addAction(auth_action)

        output_folder_action = QAction('Open &Output Folder', self)
        output_folder_action.setShortcut('Ctrl+O')
        output_folder_action.setStatusTip(
            'Open the output folder in file explorer')
        output_folder_action.triggered.connect(self.open_output_folder)
        file_menu.addAction(output_folder_action)

        file_menu.addSeparator()

        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit the application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menubar.addMenu('&Help')

        docs_action = QAction('&Documentation', self)
        docs_action.setShortcut('F1')
        docs_action.setStatusTip('Open documentation in browser')
        docs_action.triggered.connect(self.open_documentation)
        help_menu.addAction(docs_action)

        about_action = QAction('&About', self)
        about_action.setStatusTip('About Fragmenta Desktop')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def apply_menu_styling(self):

        stylesheet = """
        QMenuBar {
            background-color: #0D1117;
            border-bottom: 1px solid #30363D;
            color: #F0F6FC;
            font-family: 'JetBrains Mono', 'Space Mono', 'Courier New', monospace;
            font-size: 13px;
            font-weight: 500;
            padding: 4px 8px;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 6px 12px;
            border-radius: 4px;
            margin: 2px;
            color: #F0F6FC;
        }
        
        QMenuBar::item:selected {
            background-color: #FF6B35;
            color: #FFFFFF;
        }
        
        QMenuBar::item:pressed {
            background-color: #E55A2E;
            color: #FFFFFF;
        }
        
        QMenu {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 6px;
            padding: 4px;
            color: #F0F6FC;
            font-family: 'JetBrains Mono', 'Space Mono', 'Courier New', monospace;
            font-size: 13px;
        }
        
        QMenu::item {
            padding: 8px 16px;
            border-radius: 4px;
            margin: 2px;
            color: #F0F6FC;
        }
        
        QMenu::item:selected {
            background-color: #FF6B35;
            color: #FFFFFF;
        }
        
        QMenu::separator {
            height: 1px;
            background-color: #30363D;
            margin: 4px 8px;
        }
        
        QMainWindow {
            background-color: #0D1117;

        }
        
        QWebEngineView {
            background-color: #0D1117;

            border: none;
        }
        """
        self.setStyleSheet(stylesheet)

    def show_authentication_dialog(self):

        try:
            from app.core.hf_auth_dialog import show_hf_auth_dialog_force
            self.logger.info("Opening authentication dialog...")
            show_hf_auth_dialog_force(self)
            self.logger.info("Authentication dialog closed")
        except Exception as e:
            self.logger.error(f"Error showing authentication dialog: {e}")

    def open_output_folder(self):

        try:
            import subprocess
            import platform
            from pathlib import Path

            output_path = Path("output")
            output_path.mkdir(exist_ok=True)

            system = platform.system()
            if system == "Windows":
                subprocess.run(["explorer", str(output_path.absolute())])
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(output_path.absolute())])
            else:
                subprocess.run(["xdg-open", str(output_path.absolute())])

            self.logger.info("Output folder opened successfully")
        except Exception as e:
            self.logger.error(f"Error opening output folder: {e}")

    def open_documentation(self):

        try:
            import webbrowser

            documentation_url = "https://github.com/your-repo/fragmenta-docs"
            webbrowser.open(documentation_url)

            self.logger.info("Documentation opened successfully")
        except Exception as e:
            self.logger.error(f"Error opening documentation: {e}")

    def show_about(self):

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        
        dialog = QDialog(self)
        dialog.setWindowTitle("About Fragmenta Desktop")
        dialog.setFixedSize(500, 400)
        dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 25)
        
        title_label = QLabel("Fragmenta Desktop")
        title_font = QFont("Helvetica Neue", 20, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #000000; margin-bottom: 10px;")
        
        version_label = QLabel("Version 0.1 beta")
        version_font = QFont("Helvetica Neue", 12)
        version_label.setFont(version_font)
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("color: #666666; margin-bottom: 5px;")
        
        desc_label = QLabel("An end-to-end fine-tuning pipeline for text-to-audio models.")
        desc_font = QFont("Helvetica Neue", 13)
        desc_label.setFont(desc_font)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("""
            color: #000000;
            line-height: 1.6;
            padding: 10px;
            background: transparent;
        """)
        
        copyright_label = QLabel("© 2025 Misagh Azimi\nwww.misaghazimi.com")
        copyright_font = QFont("Helvetica Neue", 11)
        copyright_label.setFont(copyright_font)
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copyright_label.setStyleSheet("color: #7f8c8d; line-height: 1.4;")
        
        tech_label = QLabel("Built with Flask, React, and PyQt6")
        tech_font = QFont("Helvetica Neue", 10)
        tech_label.setFont(tech_font)
        tech_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tech_label.setStyleSheet("color: #95a5a6; font-style: italic;")
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.setFixedSize(100, 35)
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                font-family: 'Helvetica Neue', Arial, sans-serif;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #2980b9;
                border: 2px solid #3498db;
            }
            QPushButton:pressed {
                background-color: #21618c;
                border: 2px solid #1f5582;
            }
        """)
        
        button_layout.addWidget(ok_btn)
        button_layout.addStretch()
        
        main_layout.addWidget(title_label)
        main_layout.addWidget(version_label)
        main_layout.addWidget(desc_label)
        main_layout.addStretch()
        main_layout.addWidget(copyright_label)
        main_layout.addWidget(tech_label)
        main_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        dialog.setLayout(main_layout)
        
        dialog.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ffffff, stop: 1 #f8f9fa);
                border: 1px solid #e1e8ed;
                border-radius: 12px;
            }
        """)
        
        ok_btn.clicked.connect(dialog.accept)
        
        dialog.exec()

    def start_welcome_page_monitoring(self):

        self.welcome_monitor_timer = QTimer()
        self.welcome_monitor_timer.timeout.connect(self.check_welcome_page_status)
        self.welcome_monitor_timer.start(1000)
        self.logger.info("Started monitoring welcome page status")

    def check_welcome_page_status(self):

        try:
            import requests
            
            response = requests.get('http://127.0.0.1:5001/api/welcome-page-status', timeout=2)
            if response.status_code == 200:
                data = response.json()
                if data.get('closed', False):
                    self.logger.info("Welcome page has been closed - checking for first run")
                    self.welcome_monitor_timer.stop()
                    QTimer.singleShot(500, self.check_first_run)
        except Exception as e:
            pass

    def check_first_run(self):

        try:
            from app.core.hf_auth_dialog import should_show_auth_dialog
            
            should_show, reason = should_show_auth_dialog()
            
            if should_show:
                self.logger.info(f"Models check: {reason}")
                self.show_first_run_auth_dialog()
            else:
                self.logger.info(f"Models check: {reason}")
                
        except Exception as e:
            self.logger.error(f"Error checking models: {e}")
            self.show_first_run_auth_dialog()

    def show_first_run_auth_dialog(self):

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont, QPixmap
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Welcome to Fragmenta")
        dialog.setFixedSize(520, 320)
        dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 25)
        
        title_label = QLabel("Welcome to Fragmenta!")
        title_font = QFont("Helvetica Neue", 18, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #F0F6FC; margin-bottom: 10px;")
        
        message_label = QLabel(
            "To use Fragmenta, you need at least one base model downloaded. "
            "This requires authentication with Hugging Face.\n\n"
            "Would you like to authenticate and download models now?"
        )
        message_font = QFont("Helvetica Neue", 13)
        message_label.setFont(message_font)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setStyleSheet("""
            color: #C9D1D9;
            line-height: 1.6;
            padding: 15px;
            background: transparent;
        """)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)
        button_layout.addStretch()
        
        later_btn = QPushButton("Maybe Later")
        later_btn.setFixedSize(130, 40)
        later_btn.setStyleSheet("""
            QPushButton {
                background-color: #6E7681;
                color: #F0F6FC;
                border: 1px solid #30363D;
                border-radius: 8px;
                font-family: 'Helvetica Neue', Arial, sans-serif;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #7C8B94;
                border-color: #484F58;
                border-width: 3px;
            }
            QPushButton:pressed {
                background-color: #656C76;
                border-width: 2px;
            }
        """)
        
        download_btn = QPushButton("Download Models")
        download_btn.setFixedSize(130, 40)
        download_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a6fec;
                color: #FFFFFF;
                border: 1px solid #4078f2;
                border-radius: 8px;
                font-family: 'Helvetica Neue', Arial, sans-serif;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #4078f2;
                border-color: #5087ff;
                border-width: 3px;
            }
            QPushButton:pressed {
                background-color: #2c5aa0;
                border-width: 2px;
            }
        """)
        
        button_layout.addWidget(later_btn)
        button_layout.addWidget(download_btn)
        button_layout.addStretch()
        
        main_layout.addWidget(title_label)
        main_layout.addWidget(message_label)
        main_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        dialog.setLayout(main_layout)
        
        dialog.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #161B22, stop: 1 #0D1117);
                border: 1px solid #30363D;
                border-radius: 12px;
            }
        """)
        
        user_choice = None
        
        def on_download():
            nonlocal user_choice
            user_choice = "download"
            dialog.accept()
            
        def on_later():
            nonlocal user_choice
            user_choice = "later"
            dialog.reject()
        
        download_btn.clicked.connect(on_download)
        later_btn.clicked.connect(on_later)
        
        result = dialog.exec()
        
        if user_choice == "download":
            self.logger.info("User chose to download models on first run")
            self.show_authentication_dialog()
        else:
            self.logger.info("User chose to skip model download on first run")

    def start_flask_server(self):

        self.flask_thread = FlaskThread()
        self.flask_thread.server_ready.connect(self.on_server_ready)
        self.flask_thread.start()

    def on_server_ready(self):

        QTimer.singleShot(2000, self.load_react_app)

    def load_react_app(self):

        self.web_view.setVisible(True)

        url = QUrl("http://127.0.0.1:5001")
        self.web_view.load(url)
        self.logger.info("React app loaded in desktop window")

    def resizeEvent(self, event):

        super().resizeEvent(event)
        
        if hasattr(self, 'layout') and hasattr(self, 'margin_percentage'):
            new_size = event.size()
            new_width = new_size.width()
            new_height = new_size.height()
            
            margin_width = int(new_width * self.margin_percentage / 100)
            margin_height = int(new_height * self.margin_percentage / 100)
            
            self.layout.setContentsMargins(
                margin_width,
                margin_height,
                margin_width,
                margin_height
            )

    def closeEvent(self, event):

        self.logger.info("Shutting down Fragmenta Desktop...")

        try:
            if hasattr(self, 'flask_thread'):
                self.flask_thread.terminate()
                self.flask_thread.wait(3000)

            event.accept()
            self.logger.info("Application stopped.")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            event.accept()


def main():

    if sys.platform == "darwin":
        os.environ['QT_QPA_PLATFORM'] = 'cocoa'
        os.environ['QT_MAC_WANTS_LAYER'] = '1'
        print(f"macOS: Setting Qt platform to cocoa")
    elif sys.platform == "win32":
        os.environ['QT_QPA_PLATFORM'] = 'windows'
        print(f"Windows: Setting Qt platform to windows")
    else:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        print(f"Linux: Setting Qt platform to xcb")
    
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    
    app = QApplication(sys.argv)
    
    from PyQt6.QtGui import QGuiApplication
    print(f"Qt Platform Name: {QGuiApplication.platformName()}")
    
    app.setApplicationName("Fragmenta Desktop")
    app.setApplicationDisplayName("Fragmenta Desktop") 
    app.setApplicationVersion("0.01")
    app.setOrganizationName("MAz-Codes")
    app.setOrganizationDomain("github.com/MAz-Codes")
    
    if sys.platform == "darwin":
        app.setProperty("NSApplication.applicationBundleIdentifier", "com.misaghazimi.fragmenta")
    
    if sys.platform != "darwin":
        app.setDesktopFileName("fragmenta-desktop")
    
    icon_paths = [
        project_root / "scripts" / "fragmenta_icon_1024.png",
        project_root / "scripts" / "icon.ico",
        Path(__file__).parent.parent.parent / "scripts" / "fragmenta_icon_1024.png"
    ]
    
    icon_set = False
    for icon_path in icon_paths:
        if icon_path.exists():
            from PyQt6.QtGui import QIcon
            icon = QIcon(str(icon_path))
            app.setWindowIcon(icon)
            print(f"Set app icon from: {icon_path}")
            icon_set = True
            break
    
    if not icon_set:
        print("No app icon found in expected locations")

    window = FragmentaDesktop()
    
    window.setWindowTitle("Fragmenta Desktop")
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
    window.show()
    window.raise_()
    window.activateWindow()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
