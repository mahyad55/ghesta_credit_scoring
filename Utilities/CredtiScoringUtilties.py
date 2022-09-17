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

import warnings

warnings.filterwarnings("ignore")


# %% Main Class


def import_data(raw_data_name):
    try:
        raw_data = pd.read_excel(r'.\data\\' + raw_data_name, sheet_name=None)
        df_users = raw_data['users']
        df_ghests = raw_data['ghests']
        df_orders = raw_data['orders']
        df_fcl = raw_data['fcl']
        df_bch = raw_data['bch']

        return df_users, df_ghests, df_orders, df_fcl, df_bch

    except Exception as err:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(err).__name__, err)


# %% replace persian digits with english digits
def replace_digit(x):
    try:
        per_digit = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
        eng_digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        dict_digit = dict(zip(per_digit, eng_digit))

        for p, e in dict_digit.items():
            x = x.replace(p, e)
        return x
    except Exception as err:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(err).__name__, err)


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

    except Exception as err:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(err).__name__, err)


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

    except Exception as err:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(err).__name__, err)


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

    except Exception as err:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(err).__name__, err)


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

    except Exception as err:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(err).__name__, err)


# %% age_calculation
def age_calculation(birthdate):
    try:
        today = date.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
        return age

    except Exception as err:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(err).__name__, err)


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

    except Exception as err:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(err).__name__, err)


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
    except Exception as err:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(err).__name__, err)
