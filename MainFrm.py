from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import random
from Manager.E_Manager import E_Manager
# from .. import Manager

class MyMainWindow(QMainWindow):


    def __init__(self, parent=None):

        super(MyMainWindow, self).__init__(parent)

        self.m_centralWidget = QWidget()
        self.setCentralWidget(self.m_centralWidget)
        self.InitCentralWidget()
        self.InitManager()

    def InitCentralWidget(self):

        layout1 = QHBoxLayout()

        layout2 = QVBoxLayout()
        layout1.addLayout(layout2)

        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        layout1.addWidget(self.canvas)

        self.button1 = QPushButton("Run Trainning")
        self.button1.clicked.connect(self.OnRunTrainning)
        layout2.addWidget(self.button1)

        self.button2 = QPushButton("Random Prediction")
        layout2.addWidget(self.button2)
        self.button2.clicked.connect(self.OnRandomPrediction)

        self.m_centralWidget.setLayout(layout1)

    def InitManager(self):
        self.Mgr = E_Manager()


    def OnRunTrainning(self):
        print("Run Trainning")


    def OnRandomPrediction(self):
        print(self.Mgr.mnist)
