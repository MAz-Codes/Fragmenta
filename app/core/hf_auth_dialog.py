import webbrowser
import subprocess
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QMessageBox, QGroupBox, QProgressBar,
    QCheckBox, QTabWidget, QWidget, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QDesktopServices
from PyQt6.QtCore import QUrl
from pathlib import Path


def check_required_models_exist():
    try:
        required_models = {
            'stable-audio-open-small': [
                'stable-audio-open-small-model.safetensors'
            ],
            'stable-audio-open-1.0': [
                'stable-audio-open-model.safetensors'
            ]
        }
        
        models_dir = Path('models/pretrained')
        
        if not models_dir.exists():
            return False, "Models directory does not exist"
            
        available_models = []
        missing_models = []
        
        for model_name, expected_files in required_models.items():
            model_found = False
            
            for file_name in expected_files:
                file_path = models_dir / file_name
                if file_path.exists() and file_path.is_file():
                    model_found = True
                    break
            
            if not model_found:
                model_subdir = models_dir / model_name
                if model_subdir.exists() and model_subdir.is_dir():
                    safetensors_files = list(model_subdir.glob('*.safetensors'))
                    bin_files = list(model_subdir.glob('*.bin'))
                    model_found = len(safetensors_files) > 0 or len(bin_files) > 0
            
            if model_found:
                available_models.append(model_name)
            else:
                missing_models.append(model_name)
        
        if len(available_models) > 0:
            return True, f"Found models: {', '.join(available_models)}"
        else:
            return False, f"Missing models: {', '.join(missing_models)}"
            
    except Exception as e:
        return False, f"Error checking models: {str(e)}"


def should_show_auth_dialog():
    models_exist, message = check_required_models_exist()
    
    if not models_exist:
        return True, f"Models need to be downloaded: {message}"
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        return False, f"Models available and authenticated as: {user}"
    except Exception:
        return False, "Models available (authentication not required for existing models)"


class ModelDownloadThread(QThread):
    progress_updated = pyqtSignal(str, int, str)
    download_complete = pyqtSignal(str, bool, str)

    def __init__(self, model_ids):
        super().__init__()
        self.model_ids = model_ids

    def run(self):
        from app.core.model_manager import ModelManager
        from app.core.config import get_config

        try:
            config = get_config()
            manager = ModelManager(config)

            for model_id in self.model_ids:
                try:
                    self.progress_updated.emit(
                        model_id, 0, f"Starting download of {model_id}...")

                    def progress_callback(percent, message):
                        self.progress_updated.emit(model_id, percent, message)

                    success = manager.download_model(
                        model_id, progress_callback)

                    if success:
                        self.progress_updated.emit(
                            model_id, 100, "Download complete!")
                        self.download_complete.emit(
                            model_id, True, "Downloaded successfully")
                    else:
                        self.download_complete.emit(
                            model_id, False, "Download failed")

                except Exception as e:
                    self.download_complete.emit(
                        model_id, False, f"Error: {str(e)}")

        except Exception as e:
            for model_id in self.model_ids:
                self.download_complete.emit(
                    model_id, False, f"Setup error: {str(e)}")


class HuggingFaceAuthDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hugging Face Authentication Required")
        self.setModal(True)
        self.setMinimumSize(800, 700)

        self.step_completed = {
            0: False,
            1: False,
            2: False,
            3: False,
        }

        self.selected_models = []
        self.check_current_model_status()
        self.init_ui()

    def check_current_model_status(self):
        try:
            models_exist, message = check_required_models_exist()
            self.current_model_status = {
                'models_exist': models_exist,
                'message': message
            }
            print(f"Current model status: {message}")
        except Exception as e:
            self.current_model_status = {
                'models_exist': False,
                'message': f"Error checking models: {str(e)}"
            }
            print(f"Error checking current model status: {e}")

    def init_ui(self):
        layout = QVBoxLayout()

        header_label = QLabel("Hugging Face Authentication & Model Download")
        header_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        if hasattr(self, 'current_model_status'):
            status_text = f"Current Status: {self.current_model_status['message']}"
            status_color = "#56D364" if self.current_model_status['models_exist'] else "#DB5044"
        else:
            status_text = "Checking current model status..."
            status_color = "#9198A1"
            
        status_label = QLabel(status_text)
        status_label.setStyleSheet(f"color: {status_color}; font-weight: 500; margin: 5px;")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_label)

        desc_label = QLabel(
            "Complete all steps to authenticate with Hugging Face and download AI models.\n"
            "You must complete each step before proceeding to the next one."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)

        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.update_navigation)
        layout.addWidget(self.tab_widget)

        self.create_terms_tab()
        self.create_token_tab()
        self.create_login_tab()
        self.create_test_tab()
        self.create_download_tab()

        self.create_navigation_buttons(layout)

        self.setLayout(layout)
        self.apply_dialog_styling()
        self.update_navigation()

    def create_terms_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Step 1: Accept Model Terms")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Select which models you want to download and accept their terms.\n"
            "You need to visit each model page and click 'Accept terms'."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        models_group = QGroupBox("Available Models")
        models_layout = QVBoxLayout()

        self.model_checkboxes = {}
        models = [
            ("stable-audio-open-small", "Stable Audio Open Small",
             "https://huggingface.co/stabilityai/stable-audio-open-small", "1.68GB - Good for quick generation"),
            ("stable-audio-open-1.0", "Stable Audio Open 1.0",
             "https://huggingface.co/stabilityai/stable-audio-open-1.0", "4.85GB - Higher quality, longer audio")
        ]

        for model_id, name, url, description in models:
            model_widget = QWidget()
            model_layout = QVBoxLayout()

            checkbox = QCheckBox(f"{name}")
            checkbox.setProperty('model_id', model_id)
            checkbox.setProperty('url', url)
            checkbox.toggled.connect(self.on_model_selection_changed)
            self.model_checkboxes[model_id] = checkbox
            model_layout.addWidget(checkbox)

            desc_label = QLabel(description)
            desc_label.setStyleSheet(
                "color: gray; font-size: 11px; margin-left: 20px;")
            model_layout.addWidget(desc_label)

            open_button = QPushButton(f"Open {name} Page")
            open_button.clicked.connect(
                lambda checked, u=url: self.open_url(u))
            model_layout.addWidget(open_button)

            terms_checkbox = QCheckBox(f"I have accepted the terms for {name}")
            terms_checkbox.setProperty('model_id', model_id)
            terms_checkbox.toggled.connect(self.on_terms_acceptance_changed)
            terms_checkbox.setEnabled(False)
            model_layout.addWidget(terms_checkbox)

            model_widget.setLayout(model_layout)
            models_layout.addWidget(model_widget)

        models_group.setLayout(models_layout)
        layout.addWidget(models_group)

        self.step1_complete = QCheckBox(
            "I have selected models and accepted all required terms")
        self.step1_complete.toggled.connect(
            lambda checked: self.mark_step_complete(0, checked))
        self.step1_complete.setEnabled(False)
        layout.addWidget(self.step1_complete)

        layout.addStretch()
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Step 1: Accept Terms")

    def create_token_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Step 2: Get Your Access Token")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "You need an access token to authenticate with Hugging Face.\n"
            "This token allows the app to download models on your behalf."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        instructions = QTextEdit()
        instructions.setPlainText(
            "1. Go to https://huggingface.co/settings/tokens\n"
            "2. Click 'New token'\n"
            "3. Give it a name (e.g., 'Fragmenta Desktop')\n"
            "4. Select 'Read' role\n"
            "5. IMPORTANT: Enable 'Read access to public gated repositories'\n"
            "6. Click 'Generate token'\n"
            "7. Copy the token (you'll need it in the next step)\n"
            "8. Keep this token safe - you won't see it again!"
        )
        instructions.setMaximumHeight(150)
        layout.addWidget(instructions)

        token_button = QPushButton("Open Token Settings Page")
        token_button.clicked.connect(lambda: self.open_url(
            "https://huggingface.co/settings/tokens"))
        layout.addWidget(token_button)

        self.step2_complete = QCheckBox("I have generated my access token")
        self.step2_complete.toggled.connect(
            lambda checked: self.mark_step_complete(1, checked))
        layout.addWidget(self.step2_complete)

        layout.addStretch()
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Step 2: Get Token")

    def create_login_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Step 3: Login with Your Token")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Enter your access token below to authenticate with Hugging Face.\n"
            "This will allow the app to download models."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        token_label = QLabel("Access Token:")
        layout.addWidget(token_label)

        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_input.setPlaceholderText("hf_... (paste your token here)")
        layout.addWidget(self.token_input)

        self.login_button = QPushButton("Login with Token")
        self.login_button.clicked.connect(self.login_with_token)
        layout.addWidget(self.login_button)

        self.login_status = QLabel("")
        layout.addWidget(self.login_status)

        layout.addStretch()
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Step 3: Login")

    def create_test_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Step 4: Test Authentication")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = QLabel(
            "Click the button below to verify your authentication is working correctly."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        self.test_button = QPushButton("Test Authentication")
        self.test_button.clicked.connect(self.test_authentication)
        layout.addWidget(self.test_button)

        self.test_status = QLabel("")
        layout.addWidget(self.test_status)

        self.success_label = QLabel("")
        self.success_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(self.success_label)

        layout.addStretch()
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Step 4: Test")

    def create_download_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Step 5: Download Models")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        self.download_desc = QLabel(
            "Ready to download the selected models to models/pretrained/")
        self.download_desc.setWordWrap(True)
        layout.addWidget(self.download_desc)

        self.selected_models_list = QListWidget()
        layout.addWidget(self.selected_models_list)

        self.download_button = QPushButton("Start Download")
        self.download_button.clicked.connect(self.start_model_download)
        layout.addWidget(self.download_button)

        self.progress_area = QWidget()
        progress_layout = QVBoxLayout()
        self.progress_bars = {}
        self.progress_area.setLayout(progress_layout)
        layout.addWidget(self.progress_area)

        self.download_status = QLabel("")
        layout.addWidget(self.download_status)

        layout.addStretch()
        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Step 5: Download")

    def create_navigation_buttons(self, parent_layout):
        button_layout = QHBoxLayout()

        self.prev_button = QPushButton("← Previous")
        self.prev_button.clicked.connect(self.go_previous)
        button_layout.addWidget(self.prev_button)

        button_layout.addStretch()

        help_button = QPushButton("Help")
        help_button.clicked.connect(self.show_help)
        button_layout.addWidget(help_button)

        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(self.go_next)
        button_layout.addWidget(self.next_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        parent_layout.addLayout(button_layout)

    def on_model_selection_changed(self):
        sender = self.sender()
        model_id = sender.property('model_id')

        for widget in self.findChildren(QCheckBox):
            if widget.property('model_id') == model_id and 'accepted the terms' in widget.text():
                widget.setEnabled(sender.isChecked())
                if not sender.isChecked():
                    widget.setChecked(False)
                break

        self.update_selected_models()
        self.check_step1_completion()

    def on_terms_acceptance_changed(self):
        self.check_step1_completion()

    def check_step1_completion(self):
        selected_models = []
        all_terms_accepted = True

        for model_id, checkbox in self.model_checkboxes.items():
            if checkbox.isChecked():
                selected_models.append(model_id)
                terms_accepted = False
                for widget in self.findChildren(QCheckBox):
                    if (widget.property('model_id') == model_id and
                        'accepted the terms' in widget.text() and
                            widget.isChecked()):
                        terms_accepted = True
                        break

                if not terms_accepted:
                    all_terms_accepted = False

        can_complete = len(selected_models) > 0 and all_terms_accepted
        self.step1_complete.setEnabled(can_complete)

        if can_complete and not self.step1_complete.isChecked():
            self.step1_complete.setChecked(True)
        elif not can_complete:
            self.step1_complete.setChecked(False)

    def update_selected_models(self):
        self.selected_models = []
        for model_id, checkbox in self.model_checkboxes.items():
            if checkbox.isChecked():
                self.selected_models.append(model_id)

        self.selected_models_list.clear()
        for model_id in self.selected_models:
            item = QListWidgetItem(f"{model_id}")
            self.selected_models_list.addItem(item)

        self.download_desc.setText(
            f"Ready to download {len(self.selected_models)} selected model(s) to models/pretrained/"
        )

    def mark_step_complete(self, step_index, completed):
        self.step_completed[step_index] = completed
        self.update_navigation()

    def update_navigation(self):
        if not hasattr(self, 'prev_button'):
            return

        current_tab = self.tab_widget.currentIndex()

        self.prev_button.setEnabled(current_tab > 0)

        can_go_next = False
        if current_tab < 4:
            if current_tab == 0:
                can_go_next = self.step_completed[0]
            elif current_tab == 1:
                can_go_next = self.step_completed[1]
            elif current_tab == 2:
                can_go_next = self.step_completed[2]
            elif current_tab == 3:
                can_go_next = self.step_completed[3]

        self.next_button.setEnabled(can_go_next)

        if current_tab == 3:
            self.next_button.setText("Download →")
        else:
            self.next_button.setText("Next →")

        self.close_button.setVisible(current_tab == 4)

        for i in range(5):
            tab_enabled = True
            if i > 0 and not self.step_completed[i-1]:
                tab_enabled = False
            self.tab_widget.setTabEnabled(i, tab_enabled)

    def go_previous(self):
        current = self.tab_widget.currentIndex()
        if current > 0:
            self.tab_widget.setCurrentIndex(current - 1)
            self.update_navigation()

    def go_next(self):
        current = self.tab_widget.currentIndex()
        if current < 4:
            self.tab_widget.setCurrentIndex(current + 1)
            self.update_navigation()

    def open_url(self, url):
        try:
            webbrowser.open(url)
        except Exception as e:
            self.show_modern_message("Error", f"Could not open browser: {e}", "error")

    def login_with_token(self):
        token = self.token_input.text().strip()

        if not token:
            self.show_modern_message("Error", "Please enter your access token", "warning")
            return

        if not token.startswith("hf_"):
            self.show_modern_message("Error", "Token should start with 'hf_'", "warning")
            return

        try:
            from huggingface_hub import HfApi
            api = HfApi()

            api.token = token
            user = api.whoami()

            import subprocess
            result = subprocess.run(
                ['huggingface-cli', 'login', '--token', token],
                capture_output=True, text=True
            )

            if result.returncode == 0:
                self.login_status.setText(
                    f"Successfully logged in as: {user}")
                self.login_status.setStyleSheet(
                    "color: green; font-weight: bold;")
                self.mark_step_complete(2, True)
                self.show_modern_message(
                    "Success", f"Successfully logged in as {user}", "info")
            else:
                self.login_status.setText(f"Login failed: {result.stderr}")
                self.login_status.setStyleSheet("color: red;")

        except Exception as e:
            self.login_status.setText(f"Error: {str(e)}")
            self.login_status.setStyleSheet("color: red;")

    def test_authentication(self):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            user = api.whoami()

            self.test_status.setText(f"Authenticated as: {user}")
            self.test_status.setStyleSheet("color: green; font-weight: bold;")

            self.success_label.setText(
                "Authentication successful!\n"
                "You can now download models."
            )

            self.mark_step_complete(3, True)

        except Exception as e:
            self.test_status.setText(f"Not authenticated: {str(e)}")
            self.test_status.setStyleSheet("color: red;")
            self.success_label.setText("")

    def start_model_download(self):
        if not self.selected_models:
            self.show_modern_message(
                "Error", "No models selected for download", "warning")
            return

        self.download_button.setEnabled(False)
        self.download_button.setText("Downloading...")

        layout = self.progress_area.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

        self.progress_bars = {}
        for model_id in self.selected_models:
            label = QLabel(f"{model_id}")
            layout.addWidget(label)

            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            layout.addWidget(progress_bar)

            status_label = QLabel("Preparing...")
            layout.addWidget(status_label)

            self.progress_bars[model_id] = {
                'bar': progress_bar,
                'status': status_label
            }

        self.download_thread = ModelDownloadThread(self.selected_models)
        self.download_thread.progress_updated.connect(
            self.on_download_progress)
        self.download_thread.download_complete.connect(
            self.on_download_complete)
        self.download_thread.start()

    def on_download_progress(self, model_id, percent, message):
        if model_id in self.progress_bars:
            self.progress_bars[model_id]['bar'].setValue(percent)
            self.progress_bars[model_id]['status'].setText(message)

    def on_download_complete(self, model_id, success, message):
        if model_id in self.progress_bars:
            if success:
                self.progress_bars[model_id]['status'].setText("[SUCCESS] " + message)
            else:
                self.progress_bars[model_id]['status'].setText("[FAILED] " + message)

        all_complete = True
        for model_id in self.selected_models:
            if model_id in self.progress_bars:
                status_text = self.progress_bars[model_id]['status'].text()
                if not (status_text.startswith("[SUCCESS]") or status_text.startswith("[FAILED]")):
                    all_complete = False
                    break

        if all_complete:
            self.download_button.setEnabled(True)
            self.download_button.setText("Download Complete")
            self.download_status.setText(
                "All downloads completed! Models are ready to use.")
            self.download_status.setStyleSheet(
                "color: green; font-weight: bold;")

    def show_help(self):
        self.show_modern_help_dialog()

    def show_modern_help_dialog(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Help - Hugging Face Authentication")
        dialog.setFixedSize(580, 450)
        dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(25, 25, 25, 20)
        
        title_label = QLabel("Hugging Face Authentication Help")
        title_font = QFont("Helvetica Neue", 18, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #F0F6FC; margin-bottom: 15px;")
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #161B22;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #484F58;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6E7681;
            }
        """)
        
        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")
        content_layout = QVBoxLayout()
        content_layout.setSpacing(20)
        
        help_sections = [
            ("Why do I need to authenticate?", 
             "Stable Audio models are gated and require you to accept terms and authenticate with Hugging Face before downloading."),
            ("What is an access token?", 
             "An access token is like a password that allows the app to download models on your behalf. It's stored securely on your computer."),
            ("IMPORTANT: Token Permissions", 
             "Your token MUST have 'Read access to public gated repositories' enabled. This is required to download Stable Audio models."),
            ("Is this safe?", 
             "Yes! The token only has 'Read' permissions, meaning it can only download models. It cannot modify your account or upload anything."),
            ("What if I lose my token?", 
             "You can always generate a new token from the Hugging Face settings page."),
            ("Where are models downloaded?", 
             "Models are downloaded to the models/pretrained/ directory in your Fragmenta Desktop installation.")
        ]
        
        for question, answer in help_sections:
            q_label = QLabel(question)
            q_font = QFont("Helvetica Neue", 13, QFont.Weight.Bold)
            q_label.setFont(q_font)
            q_label.setStyleSheet("color: #F0F6FC; margin-bottom: 5px;")
            
            a_label = QLabel(answer)
            a_font = QFont("Helvetica Neue", 12)
            a_label.setFont(a_font)
            a_label.setWordWrap(True)
            a_label.setStyleSheet("""
                color: #C9D1D9;
                line-height: 1.5;
                padding: 8px 15px;
                background: transparent;
                border-left: 3px solid #3a6fec;
                border-radius: 4px;
                margin-bottom: 10px;
            """)
            
            content_layout.addWidget(q_label)
            content_layout.addWidget(a_label)
        
        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("Got it!")
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
        main_layout.addWidget(scroll_area)
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
        
        ok_btn.clicked.connect(dialog.accept)
        dialog.exec()

    def show_modern_message(self, title, message, msg_type="info"):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setFixedSize(400, 200)
        dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(25, 25, 25, 20)
        
        title_label = QLabel(title)
        title_font = QFont("Helvetica Neue", 16, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if msg_type == "warning" or msg_type == "error":
            title_label.setStyleSheet("color: #DB5044; margin-bottom: 10px;")
        else:
            title_label.setStyleSheet("color: #F0F6FC; margin-bottom: 10px;")
        
        message_label = QLabel(message)
        message_font = QFont("Helvetica Neue", 12)
        message_label.setFont(message_font)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setStyleSheet("""
            color: #C9D1D9;
            line-height: 1.5;
            padding: 15px;
            background: transparent;
        """)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.setFixedSize(80, 35)
        
        if msg_type == "warning" or msg_type == "error":
            ok_btn.setStyleSheet("""
                QPushButton {
                    background-color: #DB5044;
                    color: #FFFFFF;
                    border: 1px solid #E85D75;
                    border-radius: 8px;
                    font-family: 'Helvetica Neue', Arial, sans-serif;
                    font-size: 13px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #E85D75;
                    border: 2px solid #FF6B7A;
                }
                QPushButton:pressed {
                    background-color: #C03543;
                    border: 2px solid #A02632;
                }
            """)
        else:
            ok_btn.setStyleSheet("""
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
                    border: 2px solid #5087ff;
                }
                QPushButton:pressed {
                    background-color: #2c5aa0;
                    border: 2px solid #1f4788;
                }
            """)
        
        button_layout.addWidget(ok_btn)
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
        
        ok_btn.clicked.connect(dialog.accept)
        dialog.exec()

    def apply_dialog_styling(self):
        stylesheet = """
        QDialog {
            background-color: #0D1117;
            color: #F0F6FC;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 13px;
        }
        
        QDialog::title {
            background-color: #161B22;
            color: #F0F6FC;
        }
        
        QLabel {
            color: #C9D1D9;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 13px;
        }
        
        QPushButton {
            background-color: #3a6fec;
            color: #FFFFFF;
            border: 1px solid #4078f2;
            border-radius: 6px;
            padding: 8px 16px;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 13px;
            font-weight: 500;
        }
        
        QPushButton:hover {
            background-color: #4078f2;
            border-color: #5087ff;
        }
        
        QPushButton:pressed {
            background-color: #2c5aa0;
        }
        
        QPushButton:disabled {
            background-color: #484F58;
            color: #8B949E;
            border-color: #30363D;
        }
        
        QTabWidget::pane {
            border: 1px solid #30363D;
            background-color: #161B22;
        }
        
        QTabBar::tab {
            background-color: #21262D;
            color: #C9D1D9;
            padding: 8px 16px;
            border: 1px solid #30363D;
            border-bottom: none;
            border-radius: 6px 6px 0 0;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 13px;
        }
        
        QTabBar::tab:selected {
            background-color: #161B22;
            color: #3a6fec;
            font-weight: 500;
            border-color: #30363D;
        }
        
        QTabBar::tab:hover {
            background-color: #262C36;
        }
        
        QGroupBox {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 13px;
            font-weight: 500;
            color: #F0F6FC;
            border: 1px solid #30363D;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 8px;
            background-color: #161B22;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px 0 4px;
            color: #F0F6FC;
        }
        
        QCheckBox {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 13px;
            color: #C9D1D9;
            background-color: transparent;
        }
        
        QCheckBox:disabled {
            color: #6E7681;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #30363D;
            border-radius: 4px;
            background-color: #0D1117;
        }
        
        QCheckBox::indicator:checked {
            background-color: #3a6fec;
            border: 2px solid #3a6fec;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iMTIiIHZpZXdCb3g9IjAgMCAxMiAxMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEwIDNMNC41IDguNUwyIDYiIHN0cm9rZT0iI0ZGRkZGRiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+);
        }
        
        QCheckBox::indicator:unchecked {
            background-color: #0D1117;
            border: 2px solid #30363D;
        }
        
        QCheckBox::indicator:checked:hover {
            background-color: #4078f2;
            border: 2px solid #4078f2;
        }
        
        QCheckBox::indicator:unchecked:hover {
            border: 2px solid #484F58;
            background-color: #161B22;
        }
        
        QCheckBox::indicator:disabled {
            background-color: #161B22;
            border: 2px solid #484F58;
        }
        
        QTextEdit {
            border: 1px solid #30363D;
            border-radius: 6px;
            padding: 8px;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 13px;
            background-color: #0D1117;
            color: #F0F6FC;
        }
        
        QLineEdit {
            border: 1px solid #30363D;
            border-radius: 6px;
            padding: 8px;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 13px;
            background-color: #0D1117;
            color: #F0F6FC;
        }
        
        QLineEdit:focus {
            border-color: #3a6fec;
        }
        
        QTextEdit:focus {
            border-color: #3a6fec;
        }
        
        QProgressBar {
            border: 1px solid #30363D;
            border-radius: 6px;
            text-align: center;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 13px;
            background-color: #161B22;
            color: #F0F6FC;
        }
        
        QProgressBar::chunk {
            background-color: #3a6fec;
            border-radius: 5px;
        }
        
        QListWidget {
            background-color: #0D1117;
            color: #C9D1D9;
            border: 1px solid #30363D;
            border-radius: 6px;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 13px;
        }
        
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #21262D;
        }
        
        QListWidget::item:selected {
            background-color: #3a6fec;
            color: #FFFFFF;
        }
        
        QListWidget::item:hover {
            background-color: #21262D;
        }
        
        QScrollBar:vertical {
            background-color: #161B22;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #484F58;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #6E7681;
        }
        """
        self.setStyleSheet(stylesheet)


def show_hf_auth_dialog(parent=None):
    should_show, reason = should_show_auth_dialog()
    
    if not should_show:
        print(f"Skipping authentication dialog: {reason}")
        return True
    
    print(f"Showing authentication dialog: {reason}")
    dialog = HuggingFaceAuthDialog(parent)
    return dialog.exec() == QDialog.DialogCode.Accepted


def show_hf_auth_dialog_force(parent=None):
    print("Showing authentication dialog (forced)")
    dialog = HuggingFaceAuthDialog(parent)
    return dialog.exec() == QDialog.DialogCode.Accepted
