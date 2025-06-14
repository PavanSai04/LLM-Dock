#!/usr/bin/env python3
"""
Ollama Dock - A Modern GUI for Ollama Local LLMs
Hardware-aware LLM recommendations and chat interface
"""

import sys
import json
import asyncio
import aiohttp
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QListWidget, QListWidgetItem,
    QSplitter, QFrame, QProgressBar, QDialog, QFormLayout, QComboBox,
    QSystemTrayIcon, QMenu, QMessageBox, QTabWidget, QScrollArea,
    QGroupBox, QSlider, QSpinBox, QCheckBox, QTextBrowser
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QSettings, QSize, QPropertyAnimation,
    QEasingCurve, QRect, QPoint
)
from PySide6.QtGui import (
    QFont, QIcon, QPixmap, QAction, QPalette, QColor, QLinearGradient,
    QPainter, QBrush, QTextCharFormat, QSyntaxHighlighter, QTextDocument
)

# Hardware detection imports
import psutil
import platform
import subprocess
import cpuinfo
import GPUtil
from threading import Thread
import time
import requests

class HardwareDetector:
    """Detects system hardware and provides LLM recommendations"""
    
    def __init__(self):
        self.system_info = {}
        self.gpu_info = {}
        self.recommendations = []
        
    def detect_hardware(self) -> Dict:
        """Comprehensive hardware detection"""
        try:
            # Basic system info
            self.system_info = {
                'os': platform.system(),
                'os_version': platform.version(),
                'architecture': platform.architecture()[0],
                'processor': platform.processor(),
                'cpu_count': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'total_ram': psutil.virtual_memory().total,
                'available_ram': psutil.virtual_memory().available,
                'disk_space': psutil.disk_usage('/').free if platform.system() != 'Windows' else psutil.disk_usage('C:').free
            }
            
            # Detailed CPU info
            try:
                cpu_info = cpuinfo.get_cpu_info()
                self.system_info.update({
                    'cpu_brand': cpu_info.get('brand_raw', 'Unknown'),
                    'cpu_arch': cpu_info.get('arch', 'Unknown'),
                    'cpu_flags': cpu_info.get('flags', [])
                })
            except:
                pass
            
            # GPU Detection
            self.detect_gpu()
            
            # Generate recommendations
            self.generate_recommendations()
            
            return self.system_info
            
        except Exception as e:
            print(f"Hardware detection error: {e}")
            return {}
    
    def detect_gpu(self):
        """Detect GPU capabilities"""
        self.gpu_info = {
            'has_nvidia': False,
            'has_amd': False,
            'has_intel_graphics': False,
            'nvidia_gpus': [],
            'total_vram': 0,
            'integrated_graphics': None
        }
        
        try:
            # NVIDIA GPU detection
            nvidia_gpus = GPUtil.getGPUs()
            if nvidia_gpus:
                self.gpu_info['has_nvidia'] = True
                self.gpu_info['nvidia_gpus'] = [
                    {
                        'name': gpu.name,
                        'memory': gpu.memoryTotal,
                        'driver': gpu.driver
                    } for gpu in nvidia_gpus
                ]
                self.gpu_info['total_vram'] = sum(gpu.memoryTotal for gpu in nvidia_gpus)
        except:
            pass
        
        # Intel Graphics detection
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True)
                if 'Intel' in result.stdout:
                    self.gpu_info['has_intel_graphics'] = True
                    if 'Iris' in result.stdout:
                        self.gpu_info['integrated_graphics'] = 'Intel Iris Xe'
                    else:
                        self.gpu_info['integrated_graphics'] = 'Intel Integrated'
            else:
                # Linux/Mac detection
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if 'Intel' in result.stdout and 'VGA' in result.stdout:
                    self.gpu_info['has_intel_graphics'] = True
                    self.gpu_info['integrated_graphics'] = 'Intel Integrated'
        except:
            pass
        
        # AMD GPU detection (basic)
        try:
            if platform.system() == 'Windows':
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True)
                if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                    self.gpu_info['has_amd'] = True
        except:
            pass
    
    def generate_recommendations(self):
        """Generate LLM recommendations based on hardware"""
        ram_gb = self.system_info.get('total_ram', 0) / (1024**3)
        
        self.recommendations = []
        
        # RAM-based recommendations
        if ram_gb >= 32:
            self.recommendations.extend([
                {
                    'model': 'llama3.1:70b',
                    'name': 'Llama 3.1 70B',
                    'parameters': '70B',
                    'ram_required': '32GB+',
                    'performance': 'Excellent',
                    'use_case': 'Professional tasks, complex reasoning',
                    'quantization': 'Q4_K_M',
                    'estimated_speed': '2-5 tokens/s'
                },
                {
                    'model': 'codellama:34b',
                    'name': 'CodeLlama 34B',
                    'parameters': '34B',
                    'ram_required': '24GB+',
                    'performance': 'Excellent',
                    'use_case': 'Code generation, programming',
                    'quantization': 'Q4_K_M',
                    'estimated_speed': '3-7 tokens/s'
                }
            ])
        
        if ram_gb >= 16:
            self.recommendations.extend([
                {
                    'model': 'llama3.1:8b',
                    'name': 'Llama 3.1 8B',
                    'parameters': '8B',
                    'ram_required': '8GB+',
                    'performance': 'Very Good',
                    'use_case': 'General chat, writing, analysis',
                    'quantization': 'Q4_K_M',
                    'estimated_speed': '15-25 tokens/s'
                },
                {
                    'model': 'mistral:7b',
                    'name': 'Mistral 7B',
                    'parameters': '7B',
                    'ram_required': '6GB+',
                    'performance': 'Very Good',
                    'use_case': 'Efficient general purpose',
                    'quantization': 'Q4_K_M',
                    'estimated_speed': '20-30 tokens/s'
                }
            ])
        
        if ram_gb >= 8:
            self.recommendations.extend([
                {
                    'model': 'phi3:mini',
                    'name': 'Phi-3 Mini',
                    'parameters': '3.8B',
                    'ram_required': '4GB+',
                    'performance': 'Good',
                    'use_case': 'Lightweight tasks, fast responses',
                    'quantization': 'Q4_K_M',
                    'estimated_speed': '30-50 tokens/s'
                },
                {
                    'model': 'gemma2:2b',
                    'name': 'Gemma 2 2B',
                    'parameters': '2B',
                    'ram_required': '3GB+',
                    'performance': 'Good',
                    'use_case': 'Quick responses, basic tasks',
                    'quantization': 'Q4_K_M',
                    'estimated_speed': '40-60 tokens/s'
                }
            ])
        
        if ram_gb < 8:
            self.recommendations.extend([
                {
                    'model': 'tinyllama:1.1b',
                    'name': 'TinyLlama 1.1B',
                    'parameters': '1.1B',
                    'ram_required': '2GB+',
                    'performance': 'Basic',
                    'use_case': 'Simple tasks, very fast',
                    'quantization': 'Q4_K_M',
                    'estimated_speed': '60-100 tokens/s'
                }
            ])
        
        # GPU-specific optimizations
        if self.gpu_info.get('has_nvidia'):
            for rec in self.recommendations:
                rec['gpu_acceleration'] = 'NVIDIA CUDA'
                rec['estimated_speed'] = rec['estimated_speed'].replace('tokens/s', 'tokens/s (GPU accelerated)')
        
        elif self.gpu_info.get('has_intel_graphics'):
            for rec in self.recommendations:
                if rec['parameters'] in ['2B', '3.8B', '7B']:
                    rec['gpu_acceleration'] = 'Intel Graphics (OpenVINO)'
                    rec['intel_optimized'] = True
        
        # Sort by performance and RAM requirements
        self.recommendations.sort(key=lambda x: (
            -int(x['parameters'].replace('B', '')),
            x['ram_required']
        ))

class OllamaAPI:
    """Handles Ollama API communication"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = None
        
    async def init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            await self.init_session()
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except:
            return False
    
    async def list_models(self) -> List[Dict]:
        """List available models"""
        try:
            await self.init_session()
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('models', [])
        except:
            pass
        return []
    
    async def pull_model(self, model_name: str, progress_callback=None) -> bool:
        """Pull a model from Ollama"""
        try:
            await self.init_session()
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name}
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if progress_callback:
                            try:
                                data = json.loads(line.decode())
                                progress_callback(data)
                            except:
                                pass
                    return True
        except Exception as e:
            print(f"Error pulling model: {e}")
        return False
    
    async def generate_response(self, model: str, prompt: str, context: List = None) -> str:
        """Generate response from model"""
        try:
            await self.init_session()
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            if context:
                payload["context"] = context
                
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', '')
        except Exception as e:
            print(f"Error generating response: {e}")
        return ""

class ChatWorker(QThread):
    """Worker thread for chat operations"""
    
    response_ready = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, api: OllamaAPI, model: str, prompt: str):
        super().__init__()
        self.api = api
        self.model = model
        self.prompt = prompt
        
    def run(self):
        """Run the chat operation"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            response = loop.run_until_complete(
                self.api.generate_response(self.model, self.prompt)
            )
            
            if response:
                self.response_ready.emit(response)
            else:
                self.error_occurred.emit("No response from model")
                
        except Exception as e:
            self.error_occurred.emit(str(e))

class ModelDownloadDialog(QDialog):
    """Dialog for downloading recommended models"""
    
    def __init__(self, recommendations: List[Dict], api: OllamaAPI, parent=None):
        super().__init__(parent)
        self.recommendations = recommendations
        self.api = api
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Recommended Models")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Recommended Models for Your System")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Model list
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        for rec in self.recommendations:
            model_widget = self.create_model_widget(rec)
            scroll_layout.addWidget(model_widget)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def create_model_widget(self, rec: Dict) -> QWidget:
        """Create widget for a model recommendation"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        widget.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
                background-color: #f9f9f9;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"{rec['name']} ({rec['parameters']})")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Details
        details = QLabel(
            f"RAM Required: {rec['ram_required']} | "
            f"Performance: {rec['performance']} | "
            f"Speed: {rec['estimated_speed']}"
        )
        layout.addWidget(details)
        
        # Use case
        use_case = QLabel(f"Best for: {rec['use_case']}")
        use_case.setStyleSheet("color: #666;")
        layout.addWidget(use_case)
        
        # Download button
        btn_layout = QHBoxLayout()
        download_btn = QPushButton("Download")
        download_btn.clicked.connect(lambda: self.download_model(rec))
        btn_layout.addStretch()
        btn_layout.addWidget(download_btn)
        layout.addLayout(btn_layout)
        
        widget.setLayout(layout)
        return widget
    
    def download_model(self, rec: Dict):
        """Download a model"""
        # This would implement the actual download logic
        QMessageBox.information(self, "Download", f"Downloading {rec['name']}...")

class SystemInfoWidget(QWidget):
    """Widget displaying system information"""
    
    def __init__(self, hardware_info: Dict, gpu_info: Dict):
        super().__init__()
        self.hardware_info = hardware_info
        self.gpu_info = gpu_info
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("System Information")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # System info
        info_text = QTextBrowser()
        info_text.setMaximumHeight(200)
        
        info_html = self.generate_info_html()
        info_text.setHtml(info_html)
        
        layout.addWidget(info_text)
        self.setLayout(layout)
    
    def generate_info_html(self) -> str:
        """Generate HTML for system info"""
        ram_gb = self.hardware_info.get('total_ram', 0) / (1024**3)
        
        html = f"""
        <h3>Hardware Overview</h3>
        <p><b>OS:</b> {self.hardware_info.get('os', 'Unknown')} {self.hardware_info.get('os_version', '')}</p>
        <p><b>Processor:</b> {self.hardware_info.get('cpu_brand', 'Unknown')}</p>
        <p><b>CPU Cores:</b> {self.hardware_info.get('cpu_count', 'Unknown')} cores, {self.hardware_info.get('cpu_threads', 'Unknown')} threads</p>
        <p><b>RAM:</b> {ram_gb:.1f} GB</p>
        """
        
        if self.gpu_info.get('has_nvidia'):
            html += f"<p><b>GPU:</b> NVIDIA GPU detected ({len(self.gpu_info.get('nvidia_gpus', []))} GPUs)</p>"
        elif self.gpu_info.get('has_intel_graphics'):
            html += f"<p><b>Graphics:</b> {self.gpu_info.get('integrated_graphics', 'Intel Integrated')}</p>"
        elif self.gpu_info.get('has_amd'):
            html += "<p><b>GPU:</b> AMD GPU detected</p>"
        else:
            html += "<p><b>GPU:</b> No dedicated GPU detected</p>"
        
        return html

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.hardware_detector = HardwareDetector()
        self.ollama_api = OllamaAPI()
        self.current_model = None
        self.chat_history = []
        self.settings = QSettings('OllamaDock', 'OllamaDock')
        
        self.init_ui()
        self.setup_system_tray()
        self.detect_hardware()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Ollama Dock - Local LLM Interface")
        self.setGeometry(100, 100, 1200, 800)
        
        # Apply modern styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', monospace;
            }
            QLineEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5d61;
            }
            QListWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #404040;
                border-radius: 8px;
            }
            QLabel {
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #404040;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #404040;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create tabs
        self.tab_widget = QTabWidget()
        
        # Chat tab
        self.chat_tab = self.create_chat_tab()
        self.tab_widget.addTab(self.chat_tab, "Chat")
        
        # System info tab
        self.system_tab = QWidget()
        self.tab_widget.addTab(self.system_tab, "System Info")
        
        # Models tab
        self.models_tab = self.create_models_tab()
        self.tab_widget.addTab(self.models_tab, "Models")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_chat_tab(self) -> QWidget:
        """Create the chat interface tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setHtml("""
        <div style='color: #0d7377; font-size: 16px; font-weight: bold; margin-bottom: 20px;'>
            🤖 Welcome to Ollama Dock
        </div>
        <p>Your intelligent local LLM interface with hardware-aware model recommendations.</p>
        <p>Select a model from the dropdown and start chatting!</p>
        """)
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        
        # Model selector
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(200)
        input_layout.addWidget(self.model_selector)
        
        # Input field
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        return widget
    
    def create_models_tab(self) -> QWidget:
        """Create the models management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Title
        title = QLabel("Model Management")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Installed models
        installed_label = QLabel("Installed Models:")
        installed_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(installed_label)
        
        self.installed_models_list = QListWidget()
        layout.addWidget(self.installed_models_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh Models")
        refresh_btn.clicked.connect(self.refresh_models)
        button_layout.addWidget(refresh_btn)
        
        recommendations_btn = QPushButton("Show Recommendations")
        recommendations_btn.clicked.connect(self.show_recommendations)
        button_layout.addWidget(recommendations_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return widget
    
    def setup_system_tray(self):
        """Setup system tray icon"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)
            # You'd need to provide an icon file
            # self.tray_icon.setIcon(QIcon("icon.png"))
            
            tray_menu = QMenu()
            show_action = QAction("Show", self)
            show_action.triggered.connect(self.show)
            tray_menu.addAction(show_action)
            
            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(self.quit_application)
            tray_menu.addAction(quit_action)
            
            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.show()
    
    def detect_hardware(self):
        """Detect hardware and update UI"""
        def detection_thread():
            hardware_info = self.hardware_detector.detect_hardware()
            # Update UI in main thread
            QTimer.singleShot(0, lambda: self.on_hardware_detected(hardware_info))
        
        thread = Thread(target=detection_thread)
        thread.daemon = True
        thread.start()
    
    def on_hardware_detected(self, hardware_info: Dict):
        """Called when hardware detection is complete"""
        # Update system info tab
        if hasattr(self, 'system_tab'):
            system_info_widget = SystemInfoWidget(
                hardware_info, 
                self.hardware_detector.gpu_info
            )
            
            layout = QVBoxLayout(self.system_tab)
            layout.addWidget(system_info_widget)
            
            # Add recommendations section
            if self.hardware_detector.recommendations:
                rec_label = QLabel("Recommended Models")
                rec_label.setFont(QFont("Arial", 14, QFont.Bold))
                layout.addWidget(rec_label)
                
                for rec in self.hardware_detector.recommendations[:3]:  # Show top 3
                    rec_widget = QLabel(
                        f"• {rec['name']} - {rec['use_case']} "
                        f"(Est. {rec['estimated_speed']})"
                    )
                    layout.addWidget(rec_widget)
        
        # Update status
        ram_gb = hardware_info.get('total_ram', 0) / (1024**3)
        self.statusBar().showMessage(f"Hardware detected: {ram_gb:.1f}GB RAM")
        
        # Load models
        self.refresh_models()
    
    def refresh_models(self):
        """Refresh the list of available models"""
        def refresh_thread():
            asyncio.run(self._refresh_models_async())
        
        thread = Thread(target=refresh_thread)
        thread.daemon = True
        thread.start()
    
    async def _refresh_models_async(self):
        """Async model refresh"""
        try:
            models = await self.ollama_api.list_models()
            
            # Update UI in main thread
            QTimer.singleShot(0, lambda: self._update_model_lists(models))
            
        except Exception as e:
            QTimer.singleShot(0, lambda: self.statusBar().showMessage(f"Error: {str(e)}"))
    
    def _update_model_lists(self, models: List[Dict]):
        """Update model lists in UI"""
        self.model_selector.clear()
        self.installed_models_list.clear()
        
        for model in models:
            model_name = model.get('name', 'Unknown')
            self.model_selector.addItem(model_name)
            self.installed_models_list.addItem(model_name)
        
        if models:
            self.current_model = models[0].get('name')
            self.statusBar().showMessage(f"Loaded {len(models)} models")
        else:
            self.statusBar().showMessage("No models found. Install models first.")
    
    def show_recommendations(self):
        """Show model recommendations dialog"""
        if self.hardware_detector.recommendations:
            dialog = ModelDownloadDialog(
                self.hardware_detector.recommendations,
                self.ollama_api,
                self
            )
            dialog.exec()
        else:
            QMessageBox.information(self, "Info", "No recommendations available. Please wait for hardware detection to complete.")
    
    def send_message(self):
        """Send a message to the current model"""
        message = self.input_field.text().strip()
        if not message:
            return
        
        if not self.current_model:
            QMessageBox.warning(self, "Warning", "Please select a model first.")
            return
        
        # Add user message to chat
        self.add_message_to_chat("You", message)
        self.input_field.clear()
        
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.send_button.setText("Thinking...")
        
        # Start chat worker
        self.chat_worker = ChatWorker(self.ollama_api, self.current_model, message)
        self.chat_worker.response_ready.connect(self.on_response_ready)
        self.chat_worker.error_occurred.connect(self.on_response_error)
        self.chat_worker.start()
    
    def add_message_to_chat(self, sender: str, message: str):
        """Add a message to the chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if sender == "You":
            color = "#0d7377"
            icon = "👤"
        else:
            color = "#14a085"
            icon = "🤖"
        
        html_message = f"""
        <div style='margin: 10px 0; padding: 10px; border-left: 3px solid {color}; background-color: rgba(13, 115, 119, 0.1);'>
            <div style='color: {color}; font-weight: bold; margin-bottom: 5px;'>
                {icon} {sender} <span style='color: #888; font-size: 10px;'>{timestamp}</span>
            </div>
            <div style='color: #ffffff; line-height: 1.4;'>
                {message.replace('\n', '<br>')}
            </div>
        </div>
        """
        
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertHtml(html_message)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()
    
    def on_response_ready(self, response: str):
        """Handle when response is ready"""
        self.add_message_to_chat(self.current_model, response)
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        self.input_field.setFocus()
    
    def on_response_error(self, error: str):
        """Handle response error"""
        self.add_message_to_chat("System", f"Error: {error}")
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")
        self.input_field.setFocus()
    
    def quit_application(self):
        """Quit the application"""
        asyncio.run(self.ollama_api.close_session())
        QApplication.quit()
    
    def closeEvent(self, event):
        """Handle close event"""
        if hasattr(self, 'tray_icon') and self.tray_icon.isVisible():
            QMessageBox.information(
                self,
                "Ollama Dock",
                "Application was minimized to tray."
            )
            self.hide()
            event.ignore()
        else:
            self.quit_application()

class SplashScreen(QWidget):
    """Splash screen with hardware detection progress"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize splash screen UI"""
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setFixedSize(400, 300)
        
        # Center the splash screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
        # Styling
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1e1e1e, stop:1 #2d2d2d);
                color: white;
                border-radius: 15px;
            }
            QLabel {
                color: white;
            }
            QProgressBar {
                border: 2px solid #0d7377;
                border-radius: 8px;
                text-align: center;
                background-color: #404040;
            }
            QProgressBar::chunk {
                background-color: #0d7377;
                border-radius: 6px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Logo/Title
        title = QLabel("🤖 Ollama Dock")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("AI-Powered Local LLM Interface")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #0d7377; margin-bottom: 30px;")
        layout.addWidget(subtitle)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Version info
        version_label = QLabel("v1.0.0 - Hardware-Aware LLM Recommendations")
        version_label.setFont(QFont("Arial", 8))
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #888; margin-top: 20px;")
        layout.addWidget(version_label)
        
        self.setLayout(layout)
    
    def update_progress(self, value: int, status: str):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        self.status_label.setText(status)
        QApplication.processEvents()

class PerformanceMonitor(QWidget):
    """Real-time performance monitoring widget"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.start_monitoring()
    
    def init_ui(self):
        """Initialize performance monitor UI"""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Performance Monitor")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # CPU Usage
        self.cpu_label = QLabel("CPU: 0%")
        layout.addWidget(self.cpu_label)
        
        # RAM Usage
        self.ram_label = QLabel("RAM: 0%")
        layout.addWidget(self.ram_label)
        
        # GPU Usage (if available)
        self.gpu_label = QLabel("GPU: N/A")
        layout.addWidget(self.gpu_label)
        
        # Temperature
        self.temp_label = QLabel("CPU Temp: N/A")
        layout.addWidget(self.temp_label)
        
        self.setLayout(layout)
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_stats)
        self.monitor_timer.start(2000)  # Update every 2 seconds
    
    def update_stats(self):
        """Update performance statistics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_label.setText(f"CPU: {cpu_percent:.1f}%")
            
            # RAM usage
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            ram_used_gb = ram.used / (1024**3)
            ram_total_gb = ram.total / (1024**3)
            self.ram_label.setText(f"RAM: {ram_percent:.1f}% ({ram_used_gb:.1f}/{ram_total_gb:.1f} GB)")
            
            # GPU usage (NVIDIA only for now)
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.gpu_label.setText(f"GPU: {gpu.load*100:.1f}% | VRAM: {gpu.memoryUsed}/{gpu.memoryTotal}MB")
                else:
                    self.gpu_label.setText("GPU: N/A")
            except:
                self.gpu_label.setText("GPU: N/A")
            
            # CPU Temperature (Linux/Mac mainly)
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Try to get CPU temperature
                        for name, entries in temps.items():
                            if 'cpu' in name.lower() or 'core' in name.lower():
                                if entries:
                                    temp = entries[0].current
                                    self.temp_label.setText(f"CPU Temp: {temp:.1f}°C")
                                    break
                        else:
                            self.temp_label.setText("CPU Temp: N/A")
                    else:
                        self.temp_label.setText("CPU Temp: N/A")
                else:
                    self.temp_label.setText("CPU Temp: N/A")
            except:
                self.temp_label.setText("CPU Temp: N/A")
                
        except Exception as e:
            print(f"Performance monitoring error: {e}")

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Ollama Dock")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("OllamaDock")
    
    # Set application icon (you'd need to provide an icon file)
    # app.setWindowIcon(QIcon("icon.png"))
    
    # Show splash screen
    splash = SplashScreen()
    splash.show()
    
    # Simulate initialization progress
    splash.update_progress(20, "Detecting hardware...")
    QApplication.processEvents()
    time.sleep(1)
    
    splash.update_progress(40, "Initializing AI models...")
    QApplication.processEvents()
    time.sleep(1)
    
    splash.update_progress(60, "Setting up interface...")
    QApplication.processEvents()
    time.sleep(1)
    
    splash.update_progress(80, "Connecting to Ollama...")
    QApplication.processEvents()
    time.sleep(1)
    
    splash.update_progress(100, "Ready!")
    QApplication.processEvents()
    time.sleep(0.5)
    
    # Create and show main window
    main_window = MainWindow()
    main_window.show()
    
    # Close splash screen
    splash.close()
    
    # Handle application exit
    def cleanup():
        if hasattr(main_window, 'ollama_api'):
            asyncio.run(main_window.ollama_api.close_session())
    
    app.aboutToQuit.connect(cleanup)
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()