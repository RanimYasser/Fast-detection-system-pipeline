from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy, QStyle
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class CameraView(QWidget):
    """
    Camera preview with Start/Stop buttons inside the widget.
    """
    def __init__(self, start_callback, stop_callback):
        super().__init__()

        self.video_label = QLabel()
        # Stable preview size so the UI doesn't jump
        self.video_label.setMinimumSize(1220, 880)   # width x height (16:9)
        self.video_label.setMaximumSize(1220, 880)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #2C3238; background-color: #23282D;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Placeholder before first real frame
        placeholder = QPixmap(self.video_label.minimumSize())
        placeholder.fill(Qt.black)
        self.video_label.setPixmap(placeholder)

        # Buttons
        self.start_button = QPushButton(" Start")
        self.stop_button  = QPushButton(" Stop")

        style = self.style()
        self.start_button.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.stop_button.setIcon(style.standardIcon(QStyle.SP_MediaStop))

        for btn, hover in [(self.start_button, "#144678"), (self.stop_button, "#2C3238")]:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #2E353B;
                    color: #EAEFF3;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 6px;
                }}
                QPushButton:hover {{ background-color: {hover}; }}
            """)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        media_row = QHBoxLayout()
        media_row.setSpacing(12)
        media_row.addWidget(self.start_button)
        media_row.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(media_row)
        self.setLayout(layout)

        # Wire callbacks from main
        self.start_button.clicked.connect(start_callback)
        self.stop_button.clicked.connect(stop_callback)

    def update_frame(self, pixmap: QPixmap):
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def clear(self):
        self.video_label.clear()
        placeholder = QPixmap(self.video_label.minimumSize())
        placeholder.fill(Qt.black)
        self.video_label.setPixmap(placeholder)
