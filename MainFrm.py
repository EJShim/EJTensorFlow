from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
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
        MainLayout.addLayout(LeftLayout)

        RightLayout = QVBoxLayout()
        MainLayout.addLayout(RightLayout)

        self.renderLayout = QHBoxLayout()
        RightLayout.addLayout(self.renderLayout)

        self.m_figure = plt.Figure()
        self.m_canvas = FigureCanvas(self.m_figure)
        self.renderLayout.addWidget(self.m_canvas)

        self.m_logBox = QPlainTextEdit()
        self.m_logBox.setDisabled(True)

        font = QFont()
        font.setFamily("FreeMono")
        font.setPointSize(24)
        self.m_logBox.setFont(font)
        RightLayout.addWidget(self.m_logBox)

        self.button1 = QPushButton("Run Trainning")
        self.button1.clicked.connect(self.OnRunTrainning)
        LeftLayout.addWidget(self.button1)

        self.button2 = QPushButton("Random Prediction")
        LeftLayout.addWidget(self.button2)
        self.button2.clicked.connect(self.OnRandomPrediction)

        self.button3 = QPushButton("Save Model")
        LeftLayout.addWidget(self.button3)
        self.button3.clicked.connect(self.OnSaveModel)

        self.button4 = QPushButton("Load Model")
        LeftLayout.addWidget(self.button4)
        self.button4.clicked.connect(self.OnLoadModel)

        self.button5 = QPushButton("Load JPEG")
        LeftLayout.addWidget(self.button5)
        self.button5.clicked.connect(self.OnLoadImgNet)



    def InitManager(self):
        Mgr = E_Manager(self)
        self.Mgr = Mgr


    def OnRunTrainning(self):
        self.Mgr.RunTrainning()


    def OnRandomPrediction(self):
        #clear Canvas
        self.m_figure.clf()

        #Get Image and Run Prediction
        image, idx = self.Mgr.GetRandomImage()
        self.Mgr.RunPrediction(image)

        #Redraw Canvas
        self.m_canvas.draw()

    def OnSaveModel(self):
        self.Mgr.SaveModel()

    def OnLoadModel(self):
        self.Mgr.LoadModel()


    def OnLoadImgNet(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', "./imgSamples" ,"Image files (*.jpg *.jpeg *.gif)")
        print(fname[0])

        #Clear Figure
        self.m_figure.clf()

        #Predict Image Class
        self.Mgr.inception.PredictImage(fname[0])

        #Update figure
        self.m_canvas.draw()

    def DrawGraph(self):
        self.m_figure.clf()

        ax = self.m_figure.add_subplot(111)
        ax.axis('on')
        ax.set_title("Loss Function Visualization")
        ax.set_xlabel("iteration");
        ax.set_ylabel("Loss");
        ax.grid(True)

        ax.plot( self.Mgr.plotX, self.Mgr.plotY, 'ro-' )

        self.m_canvas.draw()




    def SetLog(self, string, clear = False):

        if clear:
            self.m_logBox.setPlainText(str(string))
        else:
            self.m_logBox.appendPlainText(str(string))
