# -*- coding: utf-8 -*-
"""
Created on %(date)s

Develop a Credit Risk Model for Ghesta

@author: Mahmood Yadegari
@email : mahyad55@gmail.com
"""

# % Import libraries
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency

from Utilities.CredtiScoringUtilties import *

import warnings
warnings.filterwarnings("ignore")


# %% Import Data
raw_data_name = 'cs_malex__1401-06-12__101531.xlsx'
data = prepare_data(raw_data_name)
