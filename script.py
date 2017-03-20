from PyQt5.QtWidgets import *
from MainFrm import MyMainWindow

import sys


app = QApplication([])

window = MyMainWindow()
window.setFixedSize(1300, 800)
window.show()
sys.exit(app.exec_())
