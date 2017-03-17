from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import random

from Manager.E_Manager import E_Manager
# from .. import Manager

class MyMainWindow(QMainWindow):


    def __init__(self, parent=None):

        super(MyMainWindow, self).__init__(parent)
        self.setWindowTitle("EJ Mnist TEST")

        self.m_centralWidget = QWidget()
        self.setCentralWidget(self.m_centralWidget)
        self.InitCentralWidget()
        self.InitManager()

    def InitCentralWidget(self):

        MainLayout = QHBoxLayout()
        self.m_centralWidget.setLayout(MainLayout)

        LeftLayout = QVBoxLayout()
        RightLayout = QVBoxLayout()

        MainLayout.addLayout(LeftLayout)
        MainLayout.addLayout(RightLayout)

        self.m_figure = plt.Figure()
        self.m_canvas = FigureCanvas(self.m_figure)
        RightLayout.addWidget(self.m_canvas)

        self.m_logBox = QPlainTextEdit()
        self.m_logBox.setDisabled(True)
        RightLayout.addWidget(self.m_logBox)

        self.button1 = QPushButton("Run Trainning")
        self.button1.clicked.connect(self.OnRunTrainning)
        LeftLayout.addWidget(self.button1)

        self.button2 = QPushButton("Random Prediction")
        LeftLayout.addWidget(self.button2)
        self.button2.clicked.connect(self.OnRandomPrediction)



    def InitManager(self):
        self.Mgr = E_Manager()


    def OnRunTrainning(self):
        self.SetLog("Run Trainning")


    def OnRandomPrediction(self):

        image, idx = self.Mgr.GetRandomImage()

        self.SetLog("Generate Random Test Image")
        plot = self.m_figure.add_subplot(111)
        plot.imshow(image)
        plot.axis('off')

        self.m_canvas.draw()



    def SetLog(self, string):
        self.m_logBox.appendPlainText(string)
