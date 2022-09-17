# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:40:14 2021

@author: user
"""

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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

warnings.filterwarnings("ignore")
from sklearn import set_config
set_config(display='diagram')
figsize=(20,9)


def autolabel(rects,ax,fontsize=12, rotation=0):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    fontsize=fontsize,
                    rotation=rotation,
                    ha='center', va='bottom')
#%% data types
def dtype_comparison(train,test):
    train_types = train.dtypes
    train_types_count = train_types.groupby(train_types).count()
    
    test_types = test.dtypes
    test_types_count = test_types.groupby(test_types).count()
    
    types = ['int64','float','object']
    
    x = np.arange(len(types))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, train_types_count, width, label='Train')
    rects2 = ax.bar(x + width/2, test_types_count, width, label='Test')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('count')
    ax.set_title('Data Type Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.legend()
    

    autolabel(rects1,ax = ax)
    autolabel(rects2,ax = ax)
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.tight_layout()
    plt.show()
    
# %% Distribution Comparison - Numerical Features
def Distribution_Numerical(train,test,numerical_features):
    fig3, axes = plt.subplots(2,3 , figsize=(30, 30), sharex=False)
    fig3.suptitle('Distribution Comparison - Continous Variables', fontsize=16)
    fig3.canvas.set_window_title('Distribution Comparison - Numerical Features')
    plt.subplots_adjust(top=0.9,bottom=0.05,left=0.05,right=0.99,hspace=0.6,wspace=0.3)
    
    for i, feature in enumerate(numerical_features):
        # sns.histplot(data=combined_data, x = feature, hue="Label",ax=axes[i%6, i//6])
        plt.subplot(2,3,i+1)
        train[feature].hist(bins=50)
        test[feature].hist(bins=50)
        plt.title(feature)
        plt.legend(['Train','Test'])
    
    # plt.legend(['Train','Test'])
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # plt.tight_layout()
    plt.show()

# %% Distribution Comparison - Categorical Features
def Distribution_Numerical(train,test,categorical_features):
    
    fig5, axes = plt.subplots(2,3 , figsize=(30, 30), sharex=False)
    fig5.suptitle('Distribution Comparison - Catagorical Variables', fontsize=16)
    fig5.canvas.set_window_title('Distribution Comparison - Categorical Features')
    plt.subplots_adjust(top=0.9,bottom=0.05,left=0.05,right=0.99,hspace=0.6,wspace=0.3)
    
    for i, feature in enumerate(categorical_features):
        # sns.histplot(data=combined_data, x = feature, hue="Label",ax=axes[i%6, i//6])
        plt.subplot(2,3,i+1)
        train[feature].hist(bins=50)
        test[feature].hist(bins=50)
        plt.title(feature)
        plt.legend(['Train','Test'])
    
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # plt.tight_layout()
    plt.show()
    
# %% Missing  Value Comparison
def Missing_Value_Comparison(train,test):
    
    null_train_count = train.isnull().sum()
    null_test_count = test.isnull().sum()
    
    null_train =[f for f in train.columns if train[f].isnull().sum()>0]
    null_test  =[f for f in test.columns  if test[f].isnull().sum()>0]
    
    null_features = list(set(null_train+null_test))
    
    null_train = null_train_count[null_features]
    null_test  = null_test_count[null_features]
    
    x = np.arange(len(null_features))  # the label locations
    width = 0.35  # the width of the bars
    fig2, ax2 = plt.subplots()
    plt.subplots_adjust(top=0.926,bottom=0.15,left=0.05,right=0.977,hspace=0.2,wspace=0.2)
    rects11 = ax2.bar(x - width/2, null_train, width, label='Train')
    rects22 = ax2.bar(x + width/2, null_test, width, label='Test')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax2.set_ylabel('count')
    ax2.set_title('Missing  Value Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(null_features,rotation='vertical')
    ax2.legend()
    
    autolabel(rects11,ax = ax2,fontsize=8, rotation=90)
    autolabel(rects22,ax = ax2,fontsize=8, rotation=90)
    
    # fig2.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    
    # plt.tight_layout()
    plt.show()
    
    fig31 = plt.subplots(1,2,figsize = figsize)
    plt.subplots_adjust(top=0.95,bottom=0.15,left=0.03,right=0.98,hspace=0.2,wspace=0.2)
    plt.subplot(1,2,1)
    sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='gray')
    plt.title('Missing Vales in Train Data')
    plt.subplot(1,2,2)
    sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='gray')
    plt.title('Missing Vales in Test Data')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

# %% Box Plot
def Box_Plot(train_original,features,features_name):
    
    if features_name == 'numerical':
        nrow = 2
        ncol = 2
    elif features_name == 'categorical':
        nrow = 2
        ncol = 3
    elif features_name == 'ordinal':
        nrow = 2
        ncol = 2
        
    target = train_original['Loan_Status']
    fig6, axes = plt.subplots(nrow,ncol , figsize=(30, 30), sharex=False)
    fig6.suptitle('Distribution Comparison '+features_name + ' Variables', fontsize=16)
    fig6.canvas.set_window_title('Distribution Comparison '+features_name + ' Variables')
    plt.subplots_adjust(top=0.9,bottom=0.05,left=0.05,right=0.99,hspace=0.6,wspace=0.3)
    
    for i, feature in enumerate(features):
        sort_list = sorted(train_original.groupby(feature)['Loan_Status'].median().items(), key= lambda x:x[1], reverse = True)
        order_list = [x[0] for x in sort_list ]
        sns.boxplot(data = train_original, x = feature, y = target, order=order_list, ax=axes[i%2, i//2])
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
