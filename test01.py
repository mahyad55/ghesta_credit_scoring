# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 12:07:24 2022

@author: R.S
"""

# Clear Screen and Variavles
from IPython import get_ipython
try:
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import sys
import os
import numpy as np
import pandas as pd
from persiantools.jdatetime import JalaliDate
from datetime import date
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, StandardScaler,Normalizer,MinMaxScaler,LabelEncoder
# %% Import data
df_users =  pd.read_excel(r'.\data\\cs_malex__1401-06-12__101531.xlsx',sheet_name='users')
df_ghests =  pd.read_excel(r'.\data\\cs_malex__1401-06-12__101531.xlsx',sheet_name='ghests')
df_orders =  pd.read_excel(r'.\data\\cs_malex__1401-06-12__101531.xlsx',sheet_name='orders')
df_fcl =  pd.read_excel(r'.\data\\cs_malex__1401-06-12__101531.xlsx',sheet_name='fcl')
df_bch =  pd.read_excel(r'.\data\\cs_malex__1401-06-12__101531.xlsx',sheet_name='bch')

# %% aclculate average of delays
ghests_grDelay = df_ghests.groupby(['User_ID','Order_ID'])['Delay Days'].mean()
ghests_grDelay = pd.DataFrame(ghests_grDelay)
ghests_grDelay.reset_index(inplace=True)
    
# %% to datetime
def toMiladi(x):
    return pd.to_datetime(JalaliDate(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])).to_gregorian())

def toJalali(x):
    return (JalaliDate(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])))

def age(birthdate):
    today = date.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age
# %%
df_users = df_users.loc[df_users['Birth Year'].notna()]                       

datetime_cols = ['Customer Registration Year','Customer Registration Month',
                 'Customer Registration Day','Birth Year','Birth Month','Birth Day']

for c in datetime_cols:
    df_users[c] = df_users[c].astype(int)
    df_users[c] = df_users[c].astype(str)
    
df_users['CustomerRegistrationDate'] = df_users['Customer Registration Year']+\
                                '-'+df_users['Customer Registration Month']+\
                                    '-'+df_users['Customer Registration Day']
                                    
df_users['BirthDate'] = df_users['Birth Year']+\
                 '-'+df_users['Birth Month']+\
                 '-'+df_users['Birth Day']
                 

df_users['CustomerRegistrationDateJalali'] = df_users['CustomerRegistrationDate'].apply(toJalali)
df_users['BirthDateJalali'] = df_users['BirthDate'].apply(toJalali)

df_users['CustomerRegistrationDateMiladi'] = df_users['CustomerRegistrationDate'].apply(toMiladi)
df_users['BirthDateMiladi'] = df_users['BirthDate'].apply(toMiladi)

for c in datetime_cols:
    del df_users[c]
    
    
df_users['Age'] = df_users['BirthDateMiladi'].apply(age)
# %%
df_orders = df_orders.loc[df_orders['Submit Year'].notna()]                       

datetime_cols = ['Submit Year','Submit Month','Submit Day',
                 'Charge Year','Charge Month','Charge Day']

for c in datetime_cols:
    df_orders[c] = df_orders[c].astype(int)
    df_orders[c] = df_orders[c].astype(str)
    
df_orders['SubmitDate'] = df_orders['Submit Year']+\
                                '-'+df_orders['Submit Month']+\
                                    '-'+df_orders['Submit Day']
                                    
df_orders['ChargeDate'] = df_orders['Charge Year']+\
                 '-'+df_orders['Charge Month']+\
                 '-'+df_orders['Charge Day']
                 

df_orders['SubmitDateJalali'] = df_orders['SubmitDate'].apply(toJalali)
df_orders['ChargeDateJalali'] = df_orders['ChargeDate'].apply(toJalali)

df_orders['SubmitDateMiladi'] = df_orders['SubmitDate'].apply(toMiladi)
df_orders['ChargeDateMiladi'] = df_orders['ChargeDate'].apply(toMiladi)

for c in datetime_cols:
    del df_orders[c]
    
# %% Create a Pandas Excel writer using XlsxWriter as the engine.
# =============================================================================
# writer = pd.ExcelWriter('cs_malex__1401-06-12_edited.xlsx', engine='xlsxwriter')
# 
# # Write each dataframe to a different worksheet.
# df_users.to_excel(writer, sheet_name='df_users',index=False)
# df_ghests.to_excel(writer, sheet_name='df_ghests',index=False)
# df_orders.to_excel(writer, sheet_name='df_orders',index=False)
# df_fcl.to_excel(writer, sheet_name='df_fcl',index=False)
# df_bch.to_excel(writer, sheet_name='df_bch',index=False)
# 
# # Close the Pandas Excel writer and output the Excel file.
# writer.save()
# 
# =============================================================================

# %% Clustring
clustring_fields = ['Age','Gender','Marital_Status','Children_Number',
                    'Homeownership','Dependants','Time from Last Home Transfer',
                    'Work_Exprience_Years','Main Occupation','Main Occupation Category',
                    'Main Occupation Contract Type']

df_users_clustering = df_users[clustring_fields].copy()

numerical_features = ['Age','Children_Number','Dependants',
                    'Time from Last Home Transfer','Work_Exprience_Years']
categorical_features = ['Gender','Marital_Status','Homeownership','Main Occupation',
                        'Main Occupation Category','Main Occupation Contract Type']

# %% Fill Null Values
# X = df_users_clustering.copy()
numeric_imputer = SimpleImputer(missing_values = np.nan,strategy ='median')
categorical_imputer = SimpleImputer(missing_values = np.nan,strategy ='most_frequent')

df_users_clustering[numerical_features] = numeric_imputer.fit_transform(df_users_clustering[numerical_features])
df_users_clustering[categorical_features] = categorical_imputer.fit_transform(df_users_clustering[categorical_features])

# %% Encoding
categorical_encoder = OrdinalEncoder()
df_users_clustering[categorical_features] = categorical_encoder.fit_transform(df_users_clustering[categorical_features])


# categorical_transformer = Pipeline(steps=[
#                         ('imputer', SimpleImputer(strategy='most_frequent')),
#                         ('OrdinalEncoder', OrdinalEncoder())])

# numeric_transformer = Pipeline(steps=[
#                         ('imputer', SimpleImputer(strategy='median')),
#                         ('StandardScaler', StandardScaler())])


# # preprocessor = ColumnTransformer([
# #          ('label-encoder', categorical_transformer, categorical_features),
# #          ('standard-scaler', numeric_transformer, numerical_features)])
# preprocessor = ColumnTransformer([
#          ('label-encoder', categorical_transformer, categorical_features)])


# df_users_clustering_preprocessed_cats = preprocessor.fit_transform(df_users_clustering)
# df_users_clustering_preprocessed_cats = pd.DataFrame(df_users_clustering_preprocessed_cats,columns=categorical_features)
# df_users_clustering_preprocessed = pd.concat([df_users_clustering[numerical_features],df_users_clustering_preprocessed_cats],axis=1)





# %% clustring
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot

import plotly.io as pio
pio.renderers.default='svg'


X = np.array(df_users_clustering)


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 20), wcss,marker='o',markersize =12,markerfacecolor='red',markeredgecolor='blue',linewidth=4 )
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# %% Embeding data in 2D
import math

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_embedded = pca.fit_transform(X)
# X_embedded = TSNE(n_components=2,init='random', perplexity=3).fit_transform(X)

# %%
num_clusters = 5
cluster_model   = KMeans(n_clusters = num_clusters, init = 'k-means++', random_state = 42)
cluster_labels  = cluster_model.fit_predict(X)
cluster_centers = cluster_model.cluster_centers_


df_clusterd = pd.DataFrame(data=X_embedded,columns=['x','y'])
df_clusterd['cluster_label']=cluster_labels
df_clusterd['dist_from_center'] = 0

df_users_clustering['cluster_label'] = cluster_labels
df_users_clustering['dist_from_center'] = 0

for i,l in enumerate(list(set(cluster_labels))):
    
    df_cl = df_users_clustering.loc[df_users_clustering['cluster_label']==l]
    center_cl = cluster_centers[i,:]
    dist = []
    for r in range(len(df_cl)):
        dist.append( math.dist(df_cl.iloc[r,:-2], center_cl))

    df_users_clustering['dist_from_center'].loc[df_users_clustering['cluster_label']==l]=dist
    df_clusterd['dist_from_center'].loc[df_clusterd['cluster_label']==l]=dist

# %%

fig = px.scatter(df_clusterd, x="x", y="y", color="cluster_label",
                 size='dist_from_center', hover_data=['dist_from_center'])
# fig.show()
plot(fig)
fig.write_html('Clustring Result in 5 cluster.html')
# %% distribution 
import plotly.figure_factory as ff


group_labels  = df_users_clustering['cluster_label'].unique().tolist()
for f in numerical_features:
    
    
    # # histogram_features = ['Age','Time from Last Home Transfer', 'Work_Exprience_Years']
    # # f = 'Age'
    # fig = px.histogram(df_users_clustering, x=f, text_auto=True,marginal="box",color="cluster_label",opacity=1)
    # fig.update_layout(title_text='Distplot with Normal Distribution for '+ f)
    # plot(fig)
    
    # fig.write_html(f+' distribution Plot(with numbers).html')
    # fig.write_image(f+'_Plot.svg',format='svg')
    # fig.write_image(f+'_Plot.png',format='png',scale=5)
    # fig.write_image(f+'_Plot.jpg',format='jpg',scale=5)
    
    hist_data = []
    for l in df_users_clustering['cluster_label'].unique().tolist():
        hist_data.append(df_users_clustering[f].loc[df_users_clustering['cluster_label']==l])
    
    fig2 = ff.create_distplot(hist_data, group_labels)
    fig2.update_layout(title_text='Distplot with Normal Distribution for '+ f)
    plot(fig2)
    fig2.write_html(f+' distribution Plot(with curve).html')

# %% bar plots
from plotly.subplots import make_subplots


data = df_users[clustring_fields].copy()
histogram_features = ['Age','Children_Number', 'Dependants', 'Time from Last Home Transfer', 'Work_Exprience_Years']+categorical_features
df_uniques = data.nunique()
data = data[histogram_features].astype(str)



data[numerical_features] = numeric_imputer.fit_transform(data[numerical_features])
data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])

for f in numerical_features:
    
    data[f] = pd.to_numeric(data[f])
    data[f] = data[f].astype(int)
    data[f] = data[f].astype(str)

data['cluster_label'] = df_users_clustering['cluster_label']
data['cluster_label'] = data['cluster_label'].apply(lambda x: 'cluster_'+str(x))

for f in histogram_features:
    fig = make_subplots(rows=2, cols=3)

    for i,l in enumerate(data['cluster_label'].unique()):
        
        if i<=2:
            row=1
            col = i+1
        else:
            row=2
            col = i-2
        print(row,col)
        
        df_bar0 = data.loc[data['cluster_label']==l]
        df_bar = df_bar0.groupby([f])['cluster_label'].count()
        df_bar = pd.DataFrame(data = df_bar)
        df_bar.reset_index(inplace=True)
        df_bar.columns = [f,'count']
        
        fig.add_trace(go.Bar(x=df_bar[f], y=df_bar['count']),row=row, col=col)
        
    fig.update_layout(title_text='Bar Plot for "'+f+'" for '+l)
    plot(fig)
    fig.write_html('Bar Plot for '+f+' for '+str(l)+'.html')
    
# %% bar plots



