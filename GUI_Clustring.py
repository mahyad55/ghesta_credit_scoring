import os
import sys

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import date

# from datetime import datetime
from PyQt5.QtCore import *
import math
from PyQt5.QtWidgets import *
from PyQt5 import uic
import sip
from plotly.offline import *
from plotly.graph_objects import *
import plotly
from PyQt5.QtWebEngineWidgets import QWebEngineView
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PyQt5 import QtCore, QtGui, QtWidgets
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from PyQt5.QtWebEngineWidgets import QWebEngineView
from plotly.graph_objects import Figure, Scatter
import plotly
from PyQt5.QtWebEngineWidgets import QWebEngineView

from Utilities.CredtiScoringUtilties import *
## main code
class Contract_UI(QMainWindow,ClustringGUI):
    def __init__(self):
        super(Contract_UI, self).__init__()
        # Load the ui file
        # uic.loadUi(r".\UI\\test_ui.ui", self)
        uic.loadUi(r".\UI\\clustring_interface.ui", self)

        self.isElbowPlot = 0
        self.isClusterPlot = 0
        self.isClusterPlot_tsne = 0
        self.center_table = 0

        self.getAllElemnts()

## Initialize The App
app = QApplication(sys.argv)
UIWindow = Contract_UI()
UIWindow.show()
app.exec_()

