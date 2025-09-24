from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox, QFormLayout, QFrame
)


class LabelPanelModern(QWidget):
    """
    Left information panel: logo, batch dropdown, latest box info.
    No buttons here.
    """
    def __init__(self):
        super().__init__()
        self.setObjectName("LabelPanelModern")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 0, 12, 2)
        layout.setSpacing(14)

        # --- Logo ---
        logo = QLabel()
        try:
            logo_pixmap = QPixmap(r"C:\Users\RC-co\Desktop\Fast-detection\new_ui_app\assets\Modus Final-03.png")
            if not logo_pixmap.isNull():
                logo_pixmap = logo_pixmap.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo.setPixmap(logo_pixmap)
            else:
                logo.setText("MODUSxLABS")
                logo.setStyleSheet("color:#87D6F7; font-size:22px; font-weight:bold;")
        except Exception:
            logo.setText("MODUSxLABS")
            logo.setStyleSheet("color:#87D6F7; font-size:22px; font-weight:bold;")
        logo.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(logo, 0, Qt.AlignLeft | Qt.AlignTop)

        # --- Batch dropdown ---
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["Milk Batch", "Orange Batch", "Juhayna Batch"])
        self.batch_combo.setStyleSheet("""
    QComboBox {
        background-color: #23282D;      /* theme gray */
        color: #EAEFF3;
        padding: 10px;
        border: 1px solid #2C3238;
        border-radius: 6px;
        min-width: 270px;
    }
    QComboBox QAbstractItemView {
        background-color: #23282D;      /* popup list background */
        color: #EAEFF3;
        selection-background-color: #3A4148;
        selection-color: #FFFFFF;
    }
    QComboBox::drop-down {
        width: 26px; 
        height: 26px;
        border-left: 1px solid #2C3238;
    }
""")

        layout.addWidget(self.batch_combo, 0, Qt.AlignLeft)

        # --- Latest box info ---
        layout.addWidget(self._info_block())
        layout.addStretch(1)

        self.setLayout(layout)

    # ---------- Public API ----------
    def update_info(self, msg: dict):
        orientation_val = msg.get("orientation", msg.get("Orientation", "—"))
        self.product_type.setText(msg.get("product_type", "—"))
        self.brand.setText(msg.get("brand", "—"))
        self.flavor.setText(msg.get("flavor", "—"))
        self.capacity.setText(msg.get("capacity", "—"))
        self.barcode.setText(msg.get("barcode", "—"))
        self.orientation.setText(orientation_val)
        self.expire.setText(msg.get("expire", "—"))
        self.cap_status.setText(msg.get("cap status", msg.get("cap_status", "—")))

        status = (msg.get("status") or "—").lower()
        emoji = "✅" if status == "accepted" else "❌" if status == "rejected" else "•"
        self.status.setText(f"{emoji} {status.capitalize() if status != '—' else '—'}")

        self.reason.setText(msg.get("reason", "—"))

    def clear(self):
        for lbl in [
            self.product_type, self.brand, self.flavor, self.capacity,
            self.barcode, self.orientation, self.expire, self.cap_status,
            self.status, self.reason
        ]:
            lbl.setText("—")

    # ---------- Internals ----------
    def _lbl(self, text):
        w = QLabel(text)
        w.setStyleSheet("font-weight: bold; font-size: 16px;")
        return w

    def _val(self, text="—", minw=60):
        w = QLabel(text)
        w.setStyleSheet("color: #EAEFF3; font-size: 15px;")
        w.setMinimumWidth(minw)
        w.setMaximumWidth(560)
        w.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return w

    def _info_block(self):
        self.product_type = self._val()
        self.brand        = self._val()
        self.flavor       = self._val()
        self.capacity     = self._val()
        self.barcode      = self._val()
        self.orientation  = self._val()
        self.expire       = self._val()
        self.cap_status   = self._val()
        self.status       = self._val()
        self.reason       = self._val()

        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
            
                border: 1px solid #2C3238;
                border-radius: 10px;
                margin-bottom: 8px;
                
          
                 
            }
        """)
        form = QFormLayout(frame)
        form.setVerticalSpacing(14)
        form.setHorizontalSpacing(32)
        form.setContentsMargins(16, 16, 16, 16)

        def row(label, widget): form.addRow(self._lbl(label), widget)
        row("Product Type", self.product_type)
        row("Brand",       self.brand)
        row("Flavor",      self.flavor)
        row("Capacity",    self.capacity)
        row("Barcode",     self.barcode)
        row("Orientation", self.orientation)
        row("Expire",      self.expire)
        row("Cap Status",  self.cap_status)
        row("Status",      self.status)

        # Build a container widget which holds the info frame and the export button
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)
        container_layout.addWidget(frame)

        # Create export button and style it
        from PyQt5.QtWidgets import QPushButton
        self.export_button = QPushButton("Export excel sheet")
        self.export_button.setStyleSheet("""
            QPushButton { background-color: #D98C43; color: black; padding: 10px 20px; font-weight: bold;font-size:14px; border-radius: 6px; }
            QPushButton:hover { background-color: #B36A2C; }
        """)
        self.export_button.setFixedWidth(250)
        self.export_button.setFixedHeight(41)
        container_layout.addSpacing(390)
        
        container_layout.addStretch(1)
        
        

        

        container_layout.addWidget(self.export_button, 0, Qt.AlignLeft)

        return container
