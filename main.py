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
# data.to_excel('temp_data.xlsx',index=False)

useful_columns = ['Gender', 'Marital_Status', 'Children_Number', 'Homeownership',
                'Dependants', 'Time from Last Home Transfer', 'Work_Exprience_Years',
                'Main Occupation','Main Occupation Category', 'Main Occupation Contract Type',
                'Income or Salary','Income Spouse', 'Other Incomes', 'Mnthy_Expncs_E',
                'Mnthy_Expncs_NE', 'Cheque_Years','Stock_Edalat', 'Ins_Life', 'Ins_Supp',
                'Subcidy', 'Bourse_Code', 'ICBS_BankCsDecisionStatus','ICBS_BankCsScore',
                'ICBS_BCH_Count', 'ICBS_BCH_Sum', 'ICBS_BCH_Score', 'ICBS_FCL_TotalAmTashilat',
                'ICBS_FCL_TotalAmOriginal', 'ICBS_FCL_TotalAmBedehi', 'ICBS_FCL_TotalAmSarResid',
                'ICBS_FCL_TotalAmMoavagh', 'ICBS_FCL_TotalAmMashkuk', 'ICBS_FCL_TotalAmTahodST',
                'ICBS_FCL_Score', 'ICBS_CreditRecord_Score', 'Age', 'work_home', 'have_car',
                'Delay Days','Charge Amount', 'Customer Share', 'Ghesta Share', 'Installment Amount',
                'Repayment Installments','Due Installments', 'AmOriginal', 'AmBenefit', 'AmBedehi',
                'AmMoavagh', 'AmMashkuk','Amount_BadCheck', 'Num_Days_BadCheck']

group_columns = ['Delay Days', 'Charge Amount', 'Customer Share', 'Ghesta Share', 
                 'Installment Amount', 'Repayment Installments', 'Due Installments',
                 'AmOriginal', 'AmBenefit', 'AmBedehi', 'AmMoavagh', 'AmMashkuk',
                 'Amount_BadCheck', 'Num_Days_BadCheck']

constant_columns = list(set(useful_columns).difference(set(group_columns))) + ['User_ID']


data_sum = data.groupby(['User_ID'])[group_columns].sum()
data_sum.reset_index(inplace=True)

data_constants = data[constant_columns].copy()
data_constants.drop_duplicates(inplace=True)

data_clustring = pd.merge(data_constants,data_sum)

numeric_columns = ['Children_Number','Dependants', 'Time from Last Home Transfer', 'Work_Exprience_Years',
                'Income or Salary','Income Spouse', 'Other Incomes', 'Mnthy_Expncs_E',
                'Mnthy_Expncs_NE', 'Cheque_Years','ICBS_BankCsScore',
                'ICBS_BCH_Count', 'ICBS_BCH_Sum', 'ICBS_BCH_Score', 'ICBS_FCL_TotalAmTashilat',
                'ICBS_FCL_TotalAmOriginal', 'ICBS_FCL_TotalAmBedehi', 'ICBS_FCL_TotalAmSarResid',
                'ICBS_FCL_TotalAmMoavagh', 'ICBS_FCL_TotalAmMashkuk', 'ICBS_FCL_TotalAmTahodST',
                'ICBS_FCL_Score', 'ICBS_CreditRecord_Score', 'Age', 
                'Delay Days','Charge Amount', 'Customer Share', 'Ghesta Share', 'Installment Amount',
                'Repayment Installments','Due Installments', 'AmOriginal', 'AmBenefit', 'AmBedehi',
                'AmMoavagh', 'AmMashkuk','Amount_BadCheck', 'Num_Days_BadCheck']

categorical_columns = list(set(data_clustring.columns).difference(set(numeric_columns+['User_ID'])))

numeric_preprocessing_methods = ['Normalizer','StandardScaler','Nothing']
categorical_preprocessing_methods = ['OneHotEncoder','OrdinalEncoder']
clustring_methods = ['K-Means','K-Means++','DBSCAN','Mean-shift']



