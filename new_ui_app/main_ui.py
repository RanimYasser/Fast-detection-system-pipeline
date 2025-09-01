import sys
import multiprocessing as mp
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd

import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QMessageBox
)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from queue import Empty

from new_ui_app.pipeline_manager import PipelineManager
from new_ui_app.widgets.camera_widget import CameraWidget
from new_ui_app.widgets.label_panel import LabelPanel
from new_ui_app.widgets.header_bar import HeaderBar
from new_ui_app.style.theme import APP_STYLE
from new_ui_app.style.widgets import BUTTON_PRIMARY



class JuiceControlUI(QMainWindow):
    def __init__(self, pipeline: PipelineManager):
        super().__init__()
        self.pipeline = pipeline
        self.setWindowTitle("Juice Bottle Control and Detection System")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet(APP_STYLE)
        self.boxes=[]
        self.frame_q = mp.Queue(maxsize=2)
        self.info_q = mp.Queue(maxsize=64)
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_tick)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        main_layout.addWidget(HeaderBar())

        self.camera = CameraWidget(self.start_all, self.stop_all)
        self.labels = LabelPanel()
        self.export_btn = QPushButton("🖨 Export run excel sheet")
        self.export_btn.setStyleSheet(BUTTON_PRIMARY)
        self.export_btn.clicked.connect(self.export_excel)

        side_layout = QVBoxLayout()
        side_layout.addWidget(self.labels)
        side_layout.addStretch()
        side_layout.addWidget(self.export_btn)

        content = QHBoxLayout()
        content.addWidget(self.camera)
        content.addLayout(side_layout)

        main_layout.addLayout(content)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def start_all(self):
        try:
            selected_batch = self.labels.batch_combo.currentText()
            self.boxes = []  # reset for a fresh run
            self.pipeline.start(self.frame_q, self.info_q, batch_name=selected_batch)
            self.labels.clear()
            self.timer.start(30)
        except Exception as e:
            QMessageBox.critical(self, "Pipeline Error", str(e))


    def stop_all(self):
        self.timer.stop()
        self.pipeline.stop()
        self.camera.clear()

    def on_tick(self):
        frame = None
        for _ in range(3):
            try:
                data = self.frame_q.get_nowait()
                frame = data
            except Empty:
                break
        if frame is not None:
            arr = np.frombuffer(frame, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                self.camera.update_frame(pix)

        while True:
            try:
                evt = self.info_q.get_nowait()
            except Empty:
                break
            if not isinstance(evt, dict):
                continue

            if evt.get("event") == "box_saved":
                self.labels.update_info(evt)   # existing UI update
                # NEW: store the row for export
                self.boxes.append({
                    "id": evt.get("id"),
                    "brand": evt.get("brand"),
                    "flavor": evt.get("flavor"),
                    "capacity": evt.get("capacity"),
                    "product_type": evt.get("product_type"),
                    "barcode": evt.get("barcode"),
                    "expire": evt.get("expire"),
                    "status": evt.get("status"),
                    "reason": evt.get("reason"),
                })


    def export_excel(self):
        if not self.boxes:
            QMessageBox.information(self, "Export", "No boxes to export yet.")
            return

        df = pd.DataFrame(self.boxes)

        # Format filename with timestamp
        from datetime import datetime
        fname = f"run_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        df.to_excel(fname, index=False, engine="openpyxl")
        QMessageBox.information(self, "Export", f"Excel saved: {fname}")
        print(f"[✓] Excel file saved as {fname}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    app = QApplication(sys.argv)
    win = JuiceControlUI(PipelineManager())
    win.show()
    sys.exit(app.exec_())
