# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:25:32 2021

summery:
    Build predictive models to automate the process of targeting the right applicants.
   
steps:
    1) Import Data 
    2) Understanding the data
    3) Exploratory Data Analysis (EDA)
        i.  Univariate Analysis
        ii. Bivariate Analysis
    4) Missing value and outlier treatment
    5) Evaluation Metrics for classification problems
    6) Model Building: Part 1
    7) Logistic Regression using stratified k-folds cross-validation
    8) Feature Engineering
    9) Model Building: Part 2
        i.   Logistic Regression
        ii.  Decision Tree
        iii. Random Forest
        iv.  XGBoost
https://towardsdatascience.com/predict-loan-eligibility-using-machine-learning-models-7a14ef904057
@author: Yadegari
"""

# %%Loading Packages
# Clear screen and remove variables
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

        
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.compose import make_column_selector as selector

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")
from sklearn import set_config
set_config(display='diagram')
figsize=(20,9)
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from LoanFunctions import dtype_comparison,Distribution_Numerical
from LoanFunctions import Distribution_Numerical,Missing_Value_Comparison,Box_Plot

""" ███████████████████████████ 1) Import Data █████████████████████████████ """
#%% 
train_original = pd.read_csv('Dataset/train_ctrUa4K.csv')
train_original['Loan_Status'] = train_original['Loan_Status'].replace({'Y':0,'N':1})

train = train_original.copy()

target_name = "Loan_Status"
target = train[target_name]
target.value_counts()
train = train.drop(columns=[target_name])


test = pd.read_csv('Dataset/test_lAUu6dG.csv')

""" ███████████████████████ 2) Understanding the data ██████████████████████ """
#%% Selection based on data types
n_train_samples,n_features = train.shape
n_test_samples,n_features = test.shape
all_features = train.columns

numerical_selector = selector(dtype_exclude=object)
categorical_selector = selector(dtype_include=object)

numericals = numerical_selector(train)
categoricals = categorical_selector(train)
categoricals.remove('Loan_ID')

#%% data types
# dtype_comparison(train,test)

# %% Distribution Comparison - Numerical Features
# Distribution_Numerical(train,test,numericals)

# %% Distribution Comparison - Categorical Features
# Distribution_Numerical(train,test,categoricals)

""" ██████████████████ 3) Exploratory Data Analysis (EDA) ██████████████████ """
#%% 3) Exploratory Data Analysis (EDA)
"""Categorical features: These features have categories 
(Gender, Married, Self_Employed, Credit_History, Loan_Status)

Ordinal features: Variables in categorical features having some order involved
 (Dependents, Education, Property_Area)
 
Numerical features: These features have numerical values 
(ApplicantIncome, Co-applicantIncome, LoanAmount, Loan_Amount_Term)"""

categorical_features = ['Gender', 'Married', 'Self_Employed', 'Credit_History']
ordinal_features = ['Dependents', 'Education', 'Property_Area']
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']


categorical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('OneHotEncoder', OneHotEncoder(handle_unknown="ignore"))])

ordinal_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('OrdinalEncoder', OrdinalEncoder())])


numeric_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('StandardScaler', StandardScaler())])


# preprocessor = ColumnTransformer([
#          ('one-hot-encoder', categorical_transformer, categorical_features),
#          ('standard-scaler', numeric_transformer, numerical_features),
#          ('ordinal-encoder', ordinal_transformer, ordinal_features)])


""" ████████████████ 4) Missing value and outlier treatment ████████████████ """
# %% Missing  Value Comparison
# Missing_Value_Comparison(train,test)

# %% Box Plot
# Box_Plot(train_original,features=numerical_features,features_name='numerical')

# %% otlier 
# https://www.pluralsight.com/guides/cleaning-up-data-from-outliers

train['LoanAmount_log']=np.log(train['LoanAmount'])
test['LoanAmount_log']=np.log(test['LoanAmount'])
# train['LoanAmount_log'].hist(bins=20)

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']

train['Total_Income_log'] = np.log(train['Total_Income'])
test['Total_Income_log'] = np.log(test['Total_Income'])


train['EMI']=train['LoanAmount']/train['Loan_Amount_Term']
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']


train['Balance_Income'] = train['Total_Income']-(train['EMI']*1000)
test['Balance_Income'] = test['Total_Income']-(test['EMI']*1000)

train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)

numerical_features = ['Total_Income_log', 'LoanAmount_log', 'EMI','Balance_Income']

preprocessor = ColumnTransformer([
         ('one-hot-encoder', categorical_transformer, categorical_features),
         ('standard-scaler', numeric_transformer, numerical_features),
         ('ordinal-encoder', ordinal_transformer, ordinal_features)])
#%% Model Building
train = train.drop(['Loan_ID'],axis=1)
test = test.drop(['Loan_ID'],axis=1)

#%%
model = make_pipeline(preprocessor, XGBClassifier(objective = 'binary:logistic', model_in = 'model_1.model'))
cv_results = cross_validate(model, train,target)
scores = cv_results["test_score"]
print("The mean cross-validation accuracy is: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")
#%%