from PyQt5.QtWidgets import *

class MyMainWindow(QMainWindow):


    def __init__(self, parent=None):

        super(MyMainWindow, self).__init__(parent)

        self.m_centralWidget = QWidget()
        self.setCentralWidget(self.m_centralWidget)
        self.InitCentralWidget()

    def InitCentralWidget(self):

        layout = QVBoxLayout()

        button1 = QPushButton("button1")
        layout.addWidget(button1)

        button2 = QPushButton("button2")
        layout.addWidget(button2)

        self.m_centralWidget.setLayout(layout)
