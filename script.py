from PyQt5.QtWidgets import *

from MainFrm import MyMainWindow

import sys


app = QApplication([])

window = MyMainWindow()
# window.setFixedSize(800, 500)
window.show()
sys.exit(app.exec_())
