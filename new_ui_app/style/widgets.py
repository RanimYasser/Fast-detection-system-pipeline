# style/widgets.py

BUTTON_PRIMARY = """
    QPushButton {
        background-color: #0a2342;
        color: white;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: bold;
        border-radius: 4px;
    }
    QPushButton:hover {
        background-color: #144678;
    }
"""

COMBOBOX_STYLE = """
    QComboBox {
        background-color: #263545;
        color: white;
        padding: 6px;
        border-radius: 4px;
    }
    QComboBox QAbstractItemView {
        background-color: #1f2a3a;
        color: white;
        selection-background-color: #144678;
    }
"""

GROUPBOX_STYLE = """
    QGroupBox {
        border: none;
        font-weight: bold;
        color: #4bb3fd;
        margin-top: 10px;
        padding-top: 6px;
    }
"""

LABEL_TITLE = "color: #4bb3fd; font-weight: bold; font-size: 20px;"

LABEL_VALUE = """
    color: white;
    font-size: 14px;
    background-color: #2e2e2e;
    padding: 4px 2px;
    border-radius: 4px;
"""

VIDEO_LABEL_STYLE = "border: 2px solid #2a2a2a; background-color: #2a2a2a;"
