from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os

class HeaderBar(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()

        logo_label = QLabel()
        if os.path.exists("Cortex Logo-07.png"):
            logo = QPixmap("Cortex Logo-07.png").scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo)

        title = QLabel("Juice Boxes Control and Detection System")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white; padding-left: 12px;")

        layout.addWidget(logo_label)
        layout.addWidget(title)
        layout.addStretch()
        self.setLayout(layout)
