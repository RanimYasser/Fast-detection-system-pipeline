from PyQt5.QtWidgets import QWidget, QFormLayout, QVBoxLayout, QLabel, QGroupBox, QComboBox
from new_ui_app.style.widgets import COMBOBOX_STYLE, GROUPBOX_STYLE, LABEL_TITLE, LABEL_VALUE

class LabelPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.batch_combo = QComboBox()
        self.batch_combo.addItems(["Batch A", "Batch B", "Batch C"])
        self.batch_combo.setStyleSheet(COMBOBOX_STYLE)
        self.layout.addWidget(self._group("Batch Control", self.batch_combo))

        self.brand = self._styled_value()
        self.flavor = self._styled_value()
        self.capacity = self._styled_value()
        self.status = self._styled_value()
        self.reason = self._styled_value()

        info_layout = QFormLayout()
        for label, field in [
            ("Brand:", self.brand),
            ("Flavor:", self.flavor),
            ("Capacity:", self.capacity),
            ("Status:", self.status),
            ("Reason:", self.reason),
        ]:
            info_layout.addRow(self._styled_label(label), field)

        self.layout.addWidget(self._group("Current Box Info", info_layout))
        self.layout.addStretch()
        self.setLayout(self.layout)

    def _group(self, title, content):
        box = QGroupBox(title)
        layout = QVBoxLayout() if isinstance(content, QWidget) else content
        if isinstance(content, QWidget):
            layout.addWidget(content)
        box.setLayout(layout)
        box.setStyleSheet(GROUPBOX_STYLE)
        return box

    def _styled_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(LABEL_TITLE)
        return lbl

    def _styled_value(self):
        lbl = QLabel("—")
        lbl.setStyleSheet(LABEL_VALUE)
        lbl.setMinimumWidth(160)
        return lbl
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
