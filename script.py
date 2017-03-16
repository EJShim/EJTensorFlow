from PyQt5.QtWidgets import *

from GUI.MainFrm import MyMainWindow

import sys


app = QApplication([])
foo = MyMainWindow()
foo.show()
sys.exit(app.exec_())
