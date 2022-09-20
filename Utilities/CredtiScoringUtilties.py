# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Mahmood Yadegari
@email : mahyad55@gmail.com
"""

# % Import libraries
import numpy as np
import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

import os
import sys

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import date
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans,MeanShift,DBSCAN
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, StandardScaler,Normalizer
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# from datetime import datetime
from PyQt5.QtCore import *
import math
from PyQt5.QtWidgets import *
from PyQt5 import uic
import sip
from plotly.offline import *
from plotly.graph_objects import *
import plotly
import plotly.figure_factory as ff
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



import warnings

warnings.filterwarnings("ignore")


# %% Main Class


def import_data(raw_data_name):
    try:
        raw_data = pd.read_excel(raw_data_name, sheet_name=None)
        df_users = raw_data['users']
        df_ghests = raw_data['ghests']
        df_orders = raw_data['orders']
        df_fcl = raw_data['fcl']
        df_bch = raw_data['bch']

        return df_users, df_ghests, df_orders, df_fcl, df_bch

    except Exception as e:
        print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
            .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


# %% replace persian digits with english digits
def replace_digit(x):
    try:
        per_digit = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
        eng_digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        dict_digit = dict(zip(per_digit, eng_digit))

        for p, e in dict_digit.items():
            x = x.replace(p, e)
        return x
    except Exception as e:
        print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
            .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


# %% Convert Gregorian datetime to Jalali datetime
def gregorian_to_jalali(year, month, day, splitter='-'):
    try:
        g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

        jy = 0 if year <= 1600 else 979
        year -= 621 if year <= 1600 else 1600
        year2 = year + 1 if month > 2 else year
        days = (365 * year) + int((year2 + 3) / 4) - int((year2 + 99) / 100)
        days += int((year2 + 399) / 400) - 80 + day + g_d_m[month - 1]
        jy += 33 * int(days / 12053)
        days %= 12053
        jy += 4 * int(days / 1461)
        days %= 1461
        jy += int((days - 1) / 365)

        if days > 365:
            days = (days - 1) % 365

        if days < 186:
            jm = 1 + int(days / 31)
            jd = 1 + (days % 31)
        else:
            arit = days - 186
            jm = 7 + int(arit / 30)
            jd = 1 + (arit % 30)

        if jm < 10:
            jm = '0' + str(jm)

        if jd < 10:
            jd = '0' + str(jd)

        y = str(jy) + splitter + str(jm) + splitter + str(jd)

        return y

    except Exception as e:
        print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
            .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


# %% Convert Jalali datetime to Gregorian datetime
def jalali_to_gregorian(year, month, day, splitter='-'):
    try:
        jy = int(year)
        jm = int(month)
        jd = int(day)
        jy += 1595
        days = -355668 + (365 * jy) + ((jy // 33) * 8) + (((jy % 33) + 3) // 4) + jd
        if (jm < 7):
            days += (jm - 1) * 31
        else:
            days += ((jm - 7) * 30) + 186
        gy = 400 * (days // 146097)
        days %= 146097
        if (days > 36524):
            days -= 1
            gy += 100 * (days // 36524)
            days %= 36524
            if (days >= 365):
                days += 1
        gy += 4 * (days // 1461)
        days %= 1461

        if (days > 365):
            gy += ((days - 1) // 365)
            days = (days - 1) % 365
        gd = days + 1
        if ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)):
            kab = 29
        else:
            kab = 28
        sal_a = [0, 31, kab, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        gm = 0
        while (gm < 13 and gd > sal_a[gm]):
            gd -= sal_a[gm]
            gm += 1

        if gm < 10:
            gm = '0' + str(gm)

        if gd < 10:
            gd = '0' + str(gd)

        y = str(gy) + splitter + str(gm) + splitter + str(gd)
        y = pd.to_datetime(y)

        return y

    except Exception as e:
        print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
            .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


# %% to datetime
def toMiladi(x):
    return pd.to_datetime(jalali_to_gregorian(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])))


def toJalali(x):
    return (gregorian_to_jalali(int(x.split('-')[0]), int(x.split('-')[1]), int(x.split('-')[2])))


# %% standardize date to Normal
def To_StandardDate(d):
    try:
        d = str(d)
        d = d.encode('ascii', 'ignore').decode('utf8', 'ignore')
        try:
            sep = list(set(re.sub(r'[0-9]', '', d)))[0]
        except IndexError:
            sep = None

        if sep == None:
            if len(d) == 6:
                if d.startswith('0'):
                    d = '14' + d
                else:
                    d = '13' + d

            z = d[:4] + '-' + d[4:6] + '-' + d[6:]

        else:
            x = d.split(sep)
            if len(x[0]) == 2:
                if x[0].startswith('0'):
                    x[0] = '14' + x[0]
                else:
                    x[0] = '13' + x[0]

            if len(x[1]) == 1:
                x[1] = '0' + x[1]

            if len(x[2]) == 1:
                x[2] = '0' + x[2]

            z = x[0] + '-' + x[1] + '-' + x[2]

        return z

    except Exception as e:
        print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
            .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


# %% standardize time to (00-23) format
def To_StandardTime(Date, Time):
    try:
        Time = Time.encode('ascii', 'ignore').decode('utf8', 'ignore')
        Time_split = Time.split(':')
        Date_split = Date.split('-')
        if Time_split[0] == '24':
            Time_split[0] = '00'

            try:
                DateGregorian = str(jalali_to_gregorian(int(Date_split[0]), int(Date_split[1]), 1 + int(Date_split[2])))
                Time_standard = Time_split[0] + ':' + Time_split[1] + ':' + Time_split[2]
            except ValueError:
                DateGregorian = str(jalali_to_gregorian(int(Date_split[0]), 1 + int(Date_split[1]), 1))
                Time_standard = Time_split[0] + ':' + Time_split[1] + ':' + Time_split[2]
            except IndexError:
                Time_standard = Time_split[0] + ':' + Time_split[1] + ':00'

            except:
                DateGregorian = str(jalali_to_gregorian(1 + int(Date_split[0]), 1, 1))
                Time_standard = Time_split[0] + ':' + Time_split[1] + ':' + Time_split[2]

        else:
            try:
                DateGregorian = str(jalali_to_gregorian(int(Date_split[0]), int(Date_split[1]), int(Date_split[2])))
                Time_standard = Time_split[0] + ':' + Time_split[1] + ':' + Time_split[2]

            except IndexError:
                DateGregorian = str(jalali_to_gregorian(int(Date_split[0]), int(Date_split[1]), int(Date_split[2])))
                Time_standard = Time_split[0] + ':' + Time_split[1] + ':00'

        datetime = pd.to_datetime(DateGregorian + ' ' + Time_standard)

        return datetime, Time_standard

    except Exception as e:
        print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
            .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


# %% age_calculation
def age_calculation(birthdate):
    try:
        today = date.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age

    except Exception as e:
        print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
            .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


# %% convert a multiple columns to a datatime column
def combine_to_datatime_column(df, year_col, month_col, day_col, datetime_column_name):
    try:
        for c in [year_col, month_col, day_col]:
            df[c] = df[c].astype(int)
            df[c] = df[c].astype(str)

        df[datetime_column_name + 'Jalali'] = df[year_col] + '-' + df[month_col] + '-' + df[day_col]
        df[datetime_column_name + 'Miladi'] = df[datetime_column_name + 'Jalali'].apply(toMiladi)

        del df[year_col]
        del df[month_col]
        del df[day_col]
        return df

    except Exception as e:
        print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
            .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


# %% Prepare Data
def prepare_data(raw_data_name):
    try:
        df_users, df_ghests, df_orders, df_fcl, df_bch = import_data(raw_data_name)
        df_users = df_users.loc[df_users['Birth Year'].notna()]

        df_users = combine_to_datatime_column(df_users, 'Customer Registration Year', 'Customer Registration Month',
                                              'Customer Registration Day', 'CustomerRegistrationDate')

        df_users = combine_to_datatime_column(df_users, 'Birth Year', 'Birth Month', 'Birth Day', 'BirthDate')
        df_orders = combine_to_datatime_column(df_orders, 'Submit Year', 'Submit Month', 'Submit Day', 'SubmitDate')
        df_orders = combine_to_datatime_column(df_orders, 'Charge Year', 'Charge Month', 'Charge Day', 'ChargeDate')

        df_users['Age'] = df_users['BirthDateMiladi'].apply(age_calculation)
        df_users['work_home'] = np.where(df_users['Home_City']==df_users['Work_City'],1,0)
        df_users['have_car'] = np.where(df_users['Car_Brand'].notna(),1,0)
        ghests_grDelay = df_ghests.groupby(['User_ID', 'Order_ID'])['Delay Days'].mean()
        ghests_grDelay = pd.DataFrame(ghests_grDelay)
        ghests_grDelay.reset_index(inplace=True)

        fcl_sumAmounts = df_fcl.groupby(['User_ID'])[
            'AmOriginal', 'AmBenefit', 'AmBedehi', 'AmMoavagh', 'AmMashkuk'].sum()
        fcl_sumAmounts = pd.DataFrame(fcl_sumAmounts)
        fcl_sumAmounts.reset_index(inplace=True)

        df_bch['CheckDate'] = df_bch['CheckDate'].apply(toMiladi)
        df_bch['BackDate'] = df_bch['BackDate'].apply(toMiladi)

        df_bch['Num_Days_BadCheck'] = (df_bch['BackDate'] - df_bch['CheckDate']) / np.timedelta64(1, 'D')

        bch_sumAmounts = df_bch.groupby(['User_ID'])['Amount', 'Num_Days_BadCheck'].sum()
        bch_sumAmounts = pd.DataFrame(bch_sumAmounts)
        bch_sumAmounts.reset_index(inplace=True)
        bch_sumAmounts.rename(columns={'Amount':'Amount_BadCheck'},inplace=True)

        data = pd.merge(left=df_users, right=ghests_grDelay, how='inner',
                        left_on=['User_ID'], right_on=['User_ID'],
                        suffixes=('_users', '_ghests'))

        data = pd.merge(left=data, right=df_orders, how='inner',
                        left_on=['User_ID', 'Order_ID'], right_on=['User_ID', 'Order_ID'],
                        suffixes=('', '_orders'))

        data = pd.merge(left=data, right=fcl_sumAmounts, how='left',
                        left_on=['User_ID'], right_on=['User_ID'],
                        suffixes=('', '_fcl'))

        values = {"AmOriginal": 0, "AmBenefit": 0, "AmBedehi": 0, "AmMoavagh": 0, "AmMashkuk": 0}
        data = data.fillna(value=values)

        data = pd.merge(left=data, right=bch_sumAmounts, how='left',
                        left_on=['User_ID'], right_on=['User_ID'],
                        suffixes=('', '_bch'))

        values = {"Amount": 0, "Num_Days_BadCheck": 0}
        data = data.fillna(value=values)

        return data
    except Exception as e:
        print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
            .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


def prepare_for_clustring(data):
    try:
        useful_columns = ['Gender', 'Marital_Status', 'Children_Number', 'Homeownership',
                          'Dependants', 'Time from Last Home Transfer', 'Work_Exprience_Years',
                          'Main Occupation', 'Main Occupation Category', 'Main Occupation Contract Type',
                          'Income or Salary', 'Income Spouse', 'Other Incomes', 'Mnthy_Expncs_E',
                          'Mnthy_Expncs_NE', 'Cheque_Years', 'Stock_Edalat', 'Ins_Life', 'Ins_Supp',
                          'Subcidy', 'Bourse_Code', 'ICBS_BankCsDecisionStatus', 'ICBS_BankCsScore',
                          'ICBS_BCH_Count', 'ICBS_BCH_Sum', 'ICBS_BCH_Score', 'ICBS_FCL_TotalAmTashilat',
                          'ICBS_FCL_TotalAmOriginal', 'ICBS_FCL_TotalAmBedehi', 'ICBS_FCL_TotalAmSarResid',
                          'ICBS_FCL_TotalAmMoavagh', 'ICBS_FCL_TotalAmMashkuk', 'ICBS_FCL_TotalAmTahodST',
                          'ICBS_FCL_Score', 'ICBS_CreditRecord_Score', 'Age', 'work_home', 'have_car',
                          'Delay Days', 'Charge Amount', 'Customer Share', 'Ghesta Share', 'Installment Amount',
                          'Repayment Installments', 'Due Installments', 'AmOriginal', 'AmBenefit', 'AmBedehi',
                          'AmMoavagh', 'AmMashkuk', 'Amount_BadCheck', 'Num_Days_BadCheck']

        group_columns = ['Delay Days', 'Charge Amount', 'Customer Share', 'Ghesta Share',
                         'Installment Amount', 'Repayment Installments', 'Due Installments',
                         'AmOriginal', 'AmBenefit', 'AmBedehi', 'AmMoavagh', 'AmMashkuk',
                         'Amount_BadCheck', 'Num_Days_BadCheck']

        constant_columns = list(set(useful_columns).difference(set(group_columns))) + ['User_ID']

        data_sum = data.groupby(['User_ID'])[group_columns].sum()
        data_sum.reset_index(inplace=True)

        data_constants = data[constant_columns].copy()
        data_constants.drop_duplicates(inplace=True)

        data_clustring = pd.merge(data_constants, data_sum)

        return data_clustring

    except Exception as e:
        print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
            .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))

class ClustringGUI:

    # open Excel file with 5 sheets
    def OpenFileOffline(self):
        try:
            self.Score_orig, self.data_orig = pd.Series(), pd.DataFrame()
            fname, _ = QFileDialog.getOpenFileName(self, "Open File", ".\data\\",
                                                   "All Files (*);;Excel Files (*.xlsx)")
            suffix = fname.split('.')[-1]
            if fname:
                self.data = prepare_data(fname)
                self.data_clustring = prepare_for_clustring(self.data)
                self.comboBox_featuers.addItems(self.data_clustring.columns.tolist())

        except Exception as e:
            print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
                  .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))



    def pre_processong(self):
        try:
            items = []
            for x in range(self.listWidget.count() - 1):
                items.append(self.listWidget.item(x).text())

            numeric_method = self.comboBox_numeric.currentText()
            categorical_method = self.comboBox_categorical.currentText()
            # cluster_method = self.comboBox_cluster_method.currentText()
            # n_clusters = self.spinBox_cluster_num.value()

            All_numeric_columns = ['Children_Number', 'Dependants', 'Time from Last Home Transfer', 'Work_Exprience_Years',
                               'Income or Salary', 'Income Spouse', 'Other Incomes', 'Mnthy_Expncs_E',
                               'Mnthy_Expncs_NE', 'Cheque_Years', 'ICBS_BankCsScore',
                               'ICBS_BCH_Count', 'ICBS_BCH_Sum', 'ICBS_BCH_Score', 'ICBS_FCL_TotalAmTashilat',
                               'ICBS_FCL_TotalAmOriginal', 'ICBS_FCL_TotalAmBedehi', 'ICBS_FCL_TotalAmSarResid',
                               'ICBS_FCL_TotalAmMoavagh', 'ICBS_FCL_TotalAmMashkuk', 'ICBS_FCL_TotalAmTahodST',
                               'ICBS_FCL_Score', 'ICBS_CreditRecord_Score', 'Age',
                               'Delay Days', 'Charge Amount', 'Customer Share', 'Ghesta Share', 'Installment Amount',
                               'Repayment Installments', 'Due Installments', 'AmOriginal', 'AmBenefit', 'AmBedehi',
                               'AmMoavagh', 'AmMashkuk', 'Amount_BadCheck', 'Num_Days_BadCheck']

            All_categorical_columns = list(set(self.data_clustring.columns).difference(set(All_numeric_columns + ['User_ID'])))

            numeric_columns = list(set(All_numeric_columns).intersection(set(items)))
            categorical_columns = list(set(All_categorical_columns).intersection(set(items)))

            data_clustring = self.data_clustring[numeric_columns+categorical_columns]
            self.data_clustring_selected = data_clustring.copy()
            # %% Fill Null Values
            # X = df_users_clustering.copy()
            numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            categorical_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

            try:
                data_clustring[numeric_columns] = numeric_imputer.fit_transform(data_clustring[numeric_columns])
            except ValueError:
                pass
            try:
                data_clustring[categorical_columns] = categorical_imputer.fit_transform(data_clustring[categorical_columns])
            except ValueError:
                pass
            # %% Encoding

            if numeric_method=='Normalizer':
                self.numeric_encoder = Normalizer()
                data_clustring[numeric_columns] = self.numeric_encoder.fit_transform(data_clustring[numeric_columns])
            elif numeric_method=='StandardScaler':
                self.numeric_encoder = StandardScaler()
                data_clustring[numeric_columns] = self.numeric_encoder.fit_transform(data_clustring[numeric_columns])


            if categorical_method=='OrdinalEncoder':
                self.categorical_encoder = OrdinalEncoder()
                data_clustring[categorical_columns] =  self.categorical_encoder.fit_transform(data_clustring[categorical_columns])
            elif categorical_method=='OneHotEncoder':
                data_clustring = pd.get_dummies(data_clustring)


            self.data_clustring2 = data_clustring.copy()
            self.array_clustring = np.array(data_clustring)
            # if cluster_method=='K-Means':
            #     self.cluster_model = KMeans(n_clusters=n_clusters,init='random', random_state=42)
            # elif cluster_method=='K-Means++':
            #     self.cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
            # elif cluster_method=='DBSCAN':
            #     self.cluster_model = DBSCAN()
            # elif cluster_method=='Mean-shift':
            #     self.cluster_model = MeanShift()


        except Exception as e:
            print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
                .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


    def Elbow(self):

        try:
            self.pre_processong()
            cluster_method = self.comboBox_cluster_method.currentText()
            # n_clusters = self.spinBox_cluster_num.value()

            wcss = []
            for i in range(1, 20):
                if cluster_method == 'K-Means':
                    cluster_model = KMeans(n_clusters=i, init='random', random_state=42).fit(self.array_clustring)
                elif cluster_method == 'K-Means++':
                    cluster_model = KMeans(n_clusters=i, init='k-means++', random_state=42).fit(self.array_clustring)
                elif cluster_method == 'DBSCAN':
                    cluster_model = KMeans(n_clusters=i, init='k-means++', random_state=42).fit(self.array_clustring)
                elif cluster_method == 'Mean-shift':
                    cluster_model = KMeans(n_clusters=i, init='k-means++', random_state=42).fit(self.array_clustring)

                wcss.append(cluster_model.inertia_)


            fig_elbow = go.Figure(go.Scatter(x=np.arange(0,20),y=wcss,mode='lines+markers',))
            fig_elbow.update_layout( paper_bgcolor = '#f0f0f0', plot_bgcolor = '#f0f0f0',showlegend=False)
            fig_elbow.update_layout(title='Elbow Method for finding the Optimal Number of Clusters',
                                    xaxis = dict(title = 'Number of Clusters'),yaxis = dict(title = 'Sum of Squared Distances'),
                                    margin=dict(autoexpand=True, l=20, r=20, t=30, b=20))
            fig_elbow.update_xaxes(gridcolor='black', griddash='dash', minor_griddash="dot")
            # fig_elbow.update_yaxes(gridcolor='black', griddash='dash', minor_griddash="dot")
            # paper_bgcolor = '#f0f0f0', plot_bgcolor = '#f0f0f0',

            if self.isElbowPlot == 0:
                html_fig_elbow = '<html><body style="background-color: #f0f0f0";text-align: center;> <center>'
                html_fig_elbow+= plotly.offline.plot(fig_elbow, output_type='div', include_plotlyjs='cdn')
                html_fig_elbow += '</body></html>'
                self.plot_widget_fig_elbow = QWebEngineView()
                self.plot_widget_fig_elbow.setHtml(html_fig_elbow)
                self.verticalLayout_Elbow_plot.addWidget(self.plot_widget_fig_elbow)
                self.isElbowPlot = 1
            else:
                self.verticalLayout_Elbow_plot.removeWidget(self.plot_widget_fig_elbow)
                sip.delete(self.plot_widget_fig_elbow)
                self.plot_widget_fig_elbow = None

                html_fig_elbow = '<html><body style="background-color: #f0f0f0";text-align: center;> <center>'
                html_fig_elbow += plotly.offline.plot(fig_elbow, output_type='div', include_plotlyjs='cdn')
                html_fig_elbow += '</body></html>'
                self.plot_widget_fig_elbow = QWebEngineView()
                self.plot_widget_fig_elbow.setHtml(html_fig_elbow)
                self.verticalLayout_Elbow_plot.addWidget(self.plot_widget_fig_elbow)
                self.isElbowPlot = 1

        except Exception as e:
            print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
                .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))

    def clustring(self):
        try:
            self.pre_processong()
            X = self.array_clustring.copy()
            cluster_method = self.comboBox_cluster_method.currentText()
            n_clusters = self.spinBox_cluster_num.value()

            if cluster_method == 'K-Means':
                self.cluster_model = KMeans(n_clusters=n_clusters, init='random', random_state=42)
            elif cluster_method == 'K-Means++':
                self.cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
            elif cluster_method == 'DBSCAN':
                self.cluster_model = DBSCAN()
            elif cluster_method == 'Mean-shift':
                self.cluster_model = MeanShift()

            labels = self.cluster_model.fit_predict(self.array_clustring)
            cluster_centers = self.cluster_model.cluster_centers_
            df_clusterd = self.data_clustring2.copy()
            df_clusterd['label'] = labels

            closest, _ = pairwise_distances_argmin_min(cluster_centers, self.array_clustring)
            print(closest)
            centers = self.data_clustring_selected.copy()
            centers['label'] = labels
            centers = centers.iloc[closest]
            print(centers)


            # Embeding in 2D Dimention using PCA
            pca = PCA(n_components=2)
            X_embedded = pca.fit_transform(self.array_clustring)
            df_X_embedded_pca = pd.DataFrame(data=X_embedded, columns=['x', 'y'])
            df_X_embedded_pca['label'] = labels
            df_clusterd_pca = df_clusterd.copy()
            df_clusterd_pca['x'] = df_X_embedded_pca['x']
            df_clusterd_pca['y'] = df_X_embedded_pca['y']

            # Embeding in 2D Dimention using TSNE
            X_embedded_tsne = TSNE(n_components=2, init='random', perplexity=3).fit_transform(X)
            X_embedded_tsne = pd.DataFrame(data=X_embedded_tsne, columns=['x', 'y'])
            X_embedded_tsne['label'] = labels
            df_clusterd_tsne = df_clusterd.copy()
            df_clusterd_tsne['x'] = X_embedded_tsne['x']
            df_clusterd_tsne['y'] = X_embedded_tsne['y']


            # Bubble Chart for 2D Embeded ussing PCA
            fig_cluster = px.scatter(df_clusterd_pca,x="x",y="y",color="label",size_max=40)
            fig_cluster.update_layout(showlegend=True)
            fig_cluster.update_layout(title='Clustring Result (Embeding method : PCA)',
                                    margin=dict(autoexpand=True, l=30, r=30, t=30, b=20))
            fig_cluster.update_xaxes(visible=False, showticklabels=False)
            fig_cluster.update_yaxes(visible=False, showticklabels=False)
            fig_cluster.update_layout(paper_bgcolor='#f0f0f0', plot_bgcolor='#f0f0f0')
            if self.isClusterPlot == 0:
                html_fig_cluster = '<html><body style="background-color: #f0f0f0";text-align: center;> <center>'
                html_fig_cluster += plotly.offline.plot(fig_cluster, output_type='div', include_plotlyjs='cdn')
                html_fig_cluster += '</body></html>'
                self.plot_widget_fig_cluster = QWebEngineView()
                self.plot_widget_fig_cluster.setHtml(html_fig_cluster)
                self.verticalLayout_cluster_plot.addWidget(self.plot_widget_fig_cluster)
                self.isClusterPlot = 1
            else:
                self.verticalLayout_cluster_plot.removeWidget(self.plot_widget_fig_cluster)
                sip.delete(self.plot_widget_fig_cluster)
                self.plot_widget_fig_cluster = None

                html_fig_cluster = '<html><body style="background-color: #f0f0f0";text-align: center;> <center>'
                html_fig_cluster += plotly.offline.plot(fig_cluster, output_type='div', include_plotlyjs='cdn')
                html_fig_cluster += '</body></html>'
                self.plot_widget_fig_cluster = QWebEngineView()
                self.plot_widget_fig_cluster.setHtml(html_fig_cluster)
                self.verticalLayout_cluster_plot.addWidget(self.plot_widget_fig_cluster)
                self.isClusterPlot = 1


            # Bubble Chart for 2D Embeded ussing T-SNE
            fig_cluster_tsne = px.scatter(df_clusterd_tsne,x="x",y="y",color="label",size_max=40)
            fig_cluster_tsne.update_layout(showlegend=True)
            fig_cluster_tsne.update_layout(title='Clustring Result (Embeding method : T-SNE)',
                                    margin=dict(autoexpand=True, l=30, r=30, t=30, b=20))
            fig_cluster_tsne.update_xaxes(visible=False, showticklabels=False)
            fig_cluster_tsne.update_yaxes(visible=False, showticklabels=False)
            fig_cluster_tsne.update_layout(paper_bgcolor='#f0f0f0', plot_bgcolor='#f0f0f0')
            if self.isClusterPlot_tsne == 0:
                html_fig_cluster_tsne = '<html><body style="background-color: #f0f0f0";text-align: center;> <center>'
                html_fig_cluster_tsne += plotly.offline.plot(fig_cluster_tsne, output_type='div', include_plotlyjs='cdn')
                html_fig_cluster_tsne += '</body></html>'
                self.plot_widget_fig_cluster_tsne = QWebEngineView()
                self.plot_widget_fig_cluster_tsne.setHtml(html_fig_cluster_tsne)
                self.verticalLayout_cluster_plot_tsne.addWidget(self.plot_widget_fig_cluster_tsne)
                self.isClusterPlot_tsne= 1
            else:
                self.verticalLayout_cluster_plot_tsne.removeWidget(self.plot_widget_fig_cluster_tsne)
                sip.delete(self.plot_widget_fig_cluster_tsne)
                self.plot_widget_fig_cluster_tsne = None

                html_fig_cluster_tsne = '<html><body style="background-color: #f0f0f0";text-align: center;> <center>'
                html_fig_cluster_tsne += plotly.offline.plot(fig_cluster_tsne, output_type='div', include_plotlyjs='cdn')
                html_fig_cluster_tsne += '</body></html>'
                self.plot_widget_fig_cluster_tsne = QWebEngineView()
                self.plot_widget_fig_cluster_tsne.setHtml(html_fig_cluster_tsne)
                self.verticalLayout_cluster_plot_tsne.addWidget(self.plot_widget_fig_cluster_tsne)
                self.isClusterPlot_tsne = 1


            # create tacle of centers
            # Styled Table in Plotly
            self.centers = centers.copy()
            self.pushButton_export_centers.setEnabled(True)
            # self.label_center.setStyleSheet('''QWidget::show()''')  # it destroy the style of the objects inside the treeview widget!
            self.label_center.setHidden(False)

            fig_table = go.Figure(data=[go.Table(header=dict(values=list(centers.columns),fill_color='#282D3C',
                            align='left'),cells=dict(values=[centers[c] for c in centers.columns],fill_color='#EDF1FF',align='center'))])

            fig_table = ff.create_table(centers, height_constant=20)

            if self.center_table == 0:
                html_fig_table = '<html> <body style="background-color: #f0f0f0";text-align: center;> <center>'
                html_fig_table += plotly.offline.plot(fig_table, output_type='div', include_plotlyjs='cdn')
                html_fig_table += '</body></html>'
                self.plot_widget_fig_table = QWebEngineView()
                self.plot_widget_fig_table.setHtml(html_fig_table)
                self.verticalLayout_center_table.addWidget(self.plot_widget_fig_table)
                self.center_table= 1
            else:
                self.verticalLayout_center_table.removeWidget(self.plot_widget_fig_table)
                sip.delete(self.plot_widget_fig_table)
                self.plot_widget_fig_table = None

                html_fig_table = '<html> <body style="background-color: #f0f0f0";text-align: center;> <center>'
                html_fig_table += plotly.offline.plot(fig_table, output_type='div', include_plotlyjs='cdn')
                html_fig_table += '</body></html>'
                self.plot_widget_fig_table = QWebEngineView()
                self.plot_widget_fig_table.setHtml(html_fig_table)
                self.verticalLayout_center_table.addWidget(self.plot_widget_fig_table)
                self.center_table = 1



        except Exception as e:
            print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
                .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


    # Add Item To List
    def add_it(self):
        try:
            # Grab the item from the list box
            item = self.comboBox_featuers.currentText()

            # Add item to list
            self.listWidget.addItem(item)

            # Clear the item box
            self.comboBox_featuers.removeItem(self.comboBox_featuers.currentIndex())

        except Exception as e:
            print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
                .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


    # Delete Item From List
    def delete_it(self):
        try:
            # Grab the selected row or current row
            clicked = self.listWidget.currentRow()
            item = self.listWidget.currentItem().text()

            # Delete selected row
            self.listWidget.takeItem(clicked)
            self.comboBox_featuers.addItem(item)

        except Exception as e:
            print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
                .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))

    # Clear All Items From List
    def clear_it(self):
        try:
            self.listWidget.clear()
            self.comboBox_featuers.addItems(self.data_clustring.columns.tolist())

        except Exception as e:
            print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
                .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))

    def export_excel(self):
        try:
            self.centers.to_excel('temp_center_of_clusters.xlsx', index=False)
            # os.system('temp_data_by_clicking.csv')
            try:
                file = os.popen('temp_center_of_clusters.xlsx')
            except:
                os.system('taskkill /F /IM "EXCEL.EXE" /T')
                file = os.popen('temp_center_of_clusters.xlsx')
        except Exception as e:
            print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
                .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))


    def getAllElemnts(self):

        try:
            # Import PushButtons
            IconSize = QtCore.QSize(32, 32)
            self.pushButton_import = self.findChild(QPushButton, "pushButton_import")
            # self.pushButton_import.setIcon(QtGui.QIcon(r".\UI\icons\new\search_light.png"))
            # self.pushButton_import.setIconSize(IconSize)
            self.pushButton_Elbow = self.findChild(QPushButton, "pushButton_Elbow")
            self.pushButton_clustring = self.findChild(QPushButton, "pushButton_clustring")
            self.pushButton_add_feature = self.findChild(QPushButton, "pushButton_add_feature")
            self.pushButton_remove_feature = self.findChild(QPushButton, "pushButton_remove_feature")
            self.pushButton_remove_all_features = self.findChild(QPushButton, "pushButton_remove_all_features")
            self.pushButton_export_centers = self.findChild(QPushButton, "pushButton_export_centers")
            self.pushButton_export_centers.setEnabled(False)

            # ImportData
            self.pushButton_import.clicked.connect(self.OpenFileOffline)
            self.pushButton_add_feature.clicked.connect(self.add_it)
            self.pushButton_remove_feature.clicked.connect(self.delete_it)
            self.pushButton_remove_all_features.clicked.connect(self.clear_it)
            self.pushButton_Elbow.clicked.connect(self.Elbow)
            self.pushButton_clustring.clicked.connect(self.clustring)
            self.pushButton_export_centers.clicked.connect(self.export_excel)

            self.comboBox_featuers = self.findChild(QComboBox, "comboBox_featuers")
            self.comboBox_cluster_method = self.findChild(QComboBox, "comboBox_cluster_method")
            self.comboBox_numeric = self.findChild(QComboBox, "comboBox_numeric")
            self.comboBox_categorical = self.findChild(QComboBox, "comboBox_categorical")



            self.comboBox_cluster_method.addItems(['K-Means', 'K-Means++', 'DBSCAN', 'Mean-shift'])
            self.comboBox_categorical.addItems(['OneHotEncoder', 'OrdinalEncoder'])
            self.comboBox_numeric.addItems(['Normalizer', 'StandardScaler', 'Nothing'])

            self.spinBox_cluster_num = self.findChild(QSpinBox, "spinBox_cluster_num")


            self.listWidget = self.findChild(QListWidget, "listWidget")

            self.label_center = self.findChild(QLabel, "label_center")
            # self.label_center.setStyleSheet('''QWidget::hide()''')  # it destroy the style of the objects inside the treeview widget!
            self.label_center.setHidden(True)



        except Exception as e:
            print('Error on line: {}\nFrom File    : {}\nType of Error: {}\nThe Error    : {}\n===============================' \
                  .format(sys.exc_info()[-1].tb_lineno, __file__, type(e).__name__, e))