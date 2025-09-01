import sys, multiprocessing as mp, numpy as np, cv2, pandas as pd
from queue import Empty
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage

from new_ui_app.pipeline_manager import PipelineManager
from new_ui_app.widgets.cam_view import CameraView
from new_ui_app.widgets.label_panel import LabelPanelModern


class JuiceUI(QMainWindow):
    def __init__(self, pipeline: PipelineManager):
        super().__init__()
        self.pipeline = pipeline
        self.setWindowTitle("Juice Bottle Detection - Modern UI")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("background-color: #121417; color: #EAEFF3; font-family: Arial; font-size: 13px;")

        self.frame_q = mp.Queue(maxsize=2)
        self.info_q = mp.Queue(maxsize=64)
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_tick)
        self.boxes = []

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        row = QHBoxLayout()

        self.labels = LabelPanelModern()
        self.camera = CameraView(self.start_all, self.stop_all)

        self.labels.export_button.clicked.connect(self.export_excel)

        row.addWidget(self.labels, 2)
        row.addWidget(self.camera, 5)

        main_layout.addLayout(row)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def start_all(self):
        try:
            selected_batch = self.labels.batch_combo.currentText()
            self.boxes.clear()
            self.pipeline.start(self.frame_q, self.info_q, batch_name=selected_batch)
            self.labels.clear()
            self.timer.start(30)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def stop_all(self):
        self.timer.stop()
        self.pipeline.stop()
        self.camera.clear()

    def on_tick(self):
        frame = None
        for _ in range(3):
            try: frame = self.frame_q.get_nowait()
            except Empty: break

        if frame:
            arr = np.frombuffer(frame, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg)
                self.camera.update_frame(pix)

        while True:
            try: evt = self.info_q.get_nowait()
            except Empty: break
            if isinstance(evt, dict) and evt.get("event") == "box_saved":
                self.labels.update_info(evt)
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
            QMessageBox.information(self, "Export", "No boxes to export.")
            return
        df = pd.DataFrame(self.boxes)
        fname = f"Excel/run_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(fname, index=False, engine="openpyxl")
        QMessageBox.information(self, "Export", f"Saved: {fname}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    app = QApplication(sys.argv)
    win = JuiceUI(PipelineManager())
    win.show()
    sys.exit(app.exec_())
