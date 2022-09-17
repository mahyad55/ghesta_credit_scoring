# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:37:24 2021

summery:
    Build predictive models to automate the process of targeting the right applicants.
   
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
from sklearn.metrics import accuracy_score

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

train = train.drop('Loan_ID',axis =1)
test = test.drop('Loan_ID',axis =1)


""" ██████████████████ 3) Exploratory Data Analysis (EDA) ██████████████████ """
#%% 3) Exploratory Data Analysis (EDA)
categorical_features = ['Gender', 'Married', 'Self_Employed', 'Credit_History']

ordinal_features = ['Dependents', 'Education', 'Property_Area']

numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                      'Loan_Amount_Term']


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
#%% train and test split
x_train,x_test,y_train,y_test = train_test_split(train,target,
                                                 test_size=0.25, random_state=0)



#%% Model Building
from sklearn.linear_model import LogisticRegression

model = make_pipeline(preprocessor, LogisticRegression())

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test,y_predict)
print(f"The accuracy is: {score :.3f}")
