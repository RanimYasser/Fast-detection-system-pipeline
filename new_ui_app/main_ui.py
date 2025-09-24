import os
import sys
import cv2
import multiprocessing as mp
from datetime import datetime
from queue import Empty

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QGridLayout, QPushButton, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

from new_ui_app.pipeline_manager import PipelineManager
from new_ui_app.widgets.cam_view import CameraView
from new_ui_app.widgets.label_panel import LabelPanelModern


class JuiceUI(QMainWindow):
    def __init__(self, pipeline: PipelineManager):
        super().__init__()
        self.pipeline = pipeline
        self.setWindowTitle("Juice Bottle Detection - Modern UI")
        self.setGeometry(100, 100, 1600, 950)
        self.setStyleSheet(
            "QMainWindow { background-color: #121417; }"
            "*, QWidget { color: #EAEFF3; font-family: Arial; font-size: 13px; }"
        )

        # Queues & timer
        self.frame_q = mp.Queue(maxsize=2)
        self.info_q = mp.Queue(maxsize=64)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.boxes = []

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(16, 10, 16, 12)
        root.setSpacing(8)

        # --- Top grid: [left fixed panel | right camera] ---
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)
        grid.setContentsMargins(0, 0, 0, 0)

        self.labels = LabelPanelModern()
        self.camera = CameraView(self.start_all, self.stop_all)

        self.labels.setFixedWidth(320)
        # Connect the export button on the labels panel (if present)
        try:
            self.labels.export_button.clicked.connect(self.export_excel)
        except Exception:
            pass
        self.camera.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        label_wrap = QWidget()
        label_layout = QVBoxLayout(label_wrap)
        label_layout.setContentsMargins(0, 100, 0, 0)  # L, Top, R, Bottom → adds 40px top margin
        label_layout.addWidget(self.labels)

        grid.addWidget(label_wrap, 0, 0, alignment=Qt.AlignTop | Qt.AlignLeft)
        grid.addWidget(self.camera, 0, 1)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)

        # Center whole grid horizontally for very wide displays
        h_center = QHBoxLayout()
        h_center.addStretch(1)
        grid_wrap = QWidget()
        grid_wrap.setLayout(grid)
        h_center.addWidget(grid_wrap)
        h_center.addStretch(1)

        root.addLayout(h_center, 1)

        # --- Bottom bar: Export + (duplicate) Start/Stop ---
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 100)
        bottom.setSpacing(12)



        

      

        root.addLayout(bottom)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    # ---------- Controls ----------
    def start_all(self):
        try:
            selected_batch = self.labels.batch_combo.currentText()
            self.boxes.clear()
            self.pipeline.start(self.frame_q, self.info_q, batch_name=selected_batch)
            self.labels.clear()
            self.timer.start(30)
        except Exception as e:
            QMessageBox.critical(self, "Start Error", str(e))

    def stop_all(self):
        try:
            self.timer.stop()
            self.pipeline.stop()
            self.camera.clear()
        except Exception as e:
            QMessageBox.critical(self, "Stop Error", str(e))

    # ---------- Tick ----------
    def on_tick(self):
        # keep latest frame
        frame = None
        for _ in range(3):
            try:
                frame = self.frame_q.get_nowait()
            except Empty:
                break

        if frame:
            arr = np.frombuffer(frame, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                self.camera.update_frame(pix)

        # drain info events
        while True:
            try:
                evt = self.info_q.get_nowait()
            except Empty:
                break

            if isinstance(evt, dict) and evt.get("event") == "box_saved":
                self.labels.update_info(evt)
                self.boxes.append({
                    "product_type": evt.get("product_type", "-"),
                    "brand": evt.get("brand", "-"),
                    "flavor": evt.get("flavor", "-"),
                    "capacity": evt.get("capacity", "-"),
                    "barcode": evt.get("barcode", "-"),
                    "expire": evt.get("expire", "-"),
                    "cap_status": evt.get("cap status", evt.get("cap_status", "-")),
                    "orientation": evt.get("orientation", evt.get("Orientation", "-")),
                    "status": evt.get("status", "-"),
                    "reason": evt.get("reason", ""),
                })

    # ---------- Export ----------
    def export_excel(self):
        if not self.boxes:
            no_msg = QMessageBox(self)
            no_msg.setIcon(QMessageBox.Information)
            no_msg.setWindowTitle("Export")
            no_msg.setText("No boxes to export.")
            # Force label text color to black for visibility against dark theme
            # Force label and button text color to black and use white background
            no_msg.setStyleSheet("QMessageBox { background: white; } QLabel{ color: black; } QPushButton{ color: black; min-width: 80px; }")
            no_msg.exec_()
            return
        try:
            os.makedirs("Excel", exist_ok=True)
            path = f"Excel/run_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            pd.DataFrame(self.boxes).to_excel(path, index=False, engine="openpyxl")
            ok_msg = QMessageBox(self)
            ok_msg.setIcon(QMessageBox.Information)
            ok_msg.setWindowTitle("Export")
            ok_msg.setText(f"Saved: {path}")
            ok_msg.setStyleSheet("QMessageBox { background: white; } QLabel{ color: black; } QPushButton{ color: black; min-width: 80px; }")
            ok_msg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    app = QApplication(sys.argv)
    ui = JuiceUI(PipelineManager())
    ui.show()
    sys.exit(app.exec_())
