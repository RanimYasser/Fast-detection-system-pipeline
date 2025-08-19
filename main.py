# main.py
import sys
import multiprocessing as mp
from PyQt5.QtWidgets import QApplication
from ui_app.app import JuiceControlUI 

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    app = QApplication(sys.argv)
    win = JuiceControlUI()
    win.show()
    sys.exit(app.exec_())
