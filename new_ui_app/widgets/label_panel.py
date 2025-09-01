from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QFormLayout, QFrame
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap


class LabelPanelModern(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        layout.setContentsMargins(16, 12, 8, 0)
        layout.setSpacing(14)

        logo = QLabel()
        logo_pixmap = QPixmap(r"C:\Users\RC-co\Desktop\Fast-detection\new_ui_app\assets\Modus Final-03.png").scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo.setPixmap(logo_pixmap)
        logo.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(logo)

        layout.addWidget(self._batch_combo())
        layout.addWidget(self._info_block())
        layout.addWidget(self._export_button_container())
        layout.addStretch()

        self.setLayout(layout)

    def _batch_combo(self):
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["Milk Batch", "Orange Batch", "Batch C"])
        self.batch_combo.setStyleSheet("""
            QComboBox {
                background-color: #23282D;
                
                padding: 10px;
            }
            QComboBox::drop-down { width: 26px; height: 26px; }
        """)
        return self.batch_combo

    def _info_block(self):
        self.brand = self._val("—")
        self.flavor = self._val("—")
        self.capacity = self._val("—")
        self.status = self._val("—")
        self.reason = self._val("—")

        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #23282D;
                border: 1px solid #2C3238;
                border-radius: 8px;
            }
        """)

        form = QFormLayout(frame)
        form.setVerticalSpacing(22)
        form.setHorizontalSpacing(40)
        form.setContentsMargins(16, 16, 16, 16)

        def row(label, widget): form.addRow(self._lbl(label), widget)

        row("Brand", self.brand)
        row("Flavor", self.flavor)
        row("Capacity", self.capacity)
        row("Status", self.status)
        row("Reason", self.reason)

        return frame

    def _lbl(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #87D6F7; font-weight: bold; font-size: 18px;")
        return lbl

    def _val(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #EAEFF3; font-size: 16px;")
        lbl.setMinimumWidth(160)
        return lbl

    def _export_button_container(self):
        self.export_button = QPushButton("Export excel sheet")
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #D98C43;
                color: black;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #B36A2C; }
        """)
        wrapper = QVBoxLayout()
        wrapper.setContentsMargins(0, 700, 0, 0)
        wrapper.addWidget(self.export_button)

        container = QWidget()
        container.setLayout(wrapper)
        return container

    def update_info(self, msg: dict):
        self.brand.setText(msg.get("brand", "—"))
        self.flavor.setText(msg.get("flavor", "—"))
        self.capacity.setText(msg.get("capacity", "—"))
        status = msg.get("status", "—").lower()
        emoji = "✅" if status == "accepted" else "❌" if status == "rejected" else "•"
        self.status.setText(f"{emoji} {status.capitalize() if status != '—' else '—'}")
        self.reason.setText(msg.get("reason", "—"))

    def clear(self):
        for lbl in [self.brand, self.flavor, self.capacity, self.status, self.reason]:
            lbl.setText("—")
