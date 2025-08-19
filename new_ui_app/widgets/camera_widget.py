from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from new_ui_app.style.widgets import BUTTON_PRIMARY, VIDEO_LABEL_STYLE

class CameraWidget(QWidget):
    def __init__(self, start_callback, stop_callback):
        super().__init__()

        self.video_label = QLabel()
        self.video_label.setFixedSize(900, 600)
        self.video_label.setStyleSheet(VIDEO_LABEL_STYLE)

        self.start_button = QPushButton("▶ Start (Pipeline + Camera)")
        self.stop_button = QPushButton("■ Stop")

        self.start_button.setStyleSheet(BUTTON_PRIMARY)
        self.stop_button.setStyleSheet(BUTTON_PRIMARY)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(button_row)
        self.setLayout(layout)

        self.start_button.clicked.connect(start_callback)
        self.stop_button.clicked.connect(stop_callback)

    def update_frame(self, pixmap: QPixmap):
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def clear(self):
        self.video_label.clear()
