import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from scipy import stats
import logging
from Data_info import read_the_data
from Data_Preprocessing import preprocessing
from sklearn.preprocessing import StandardScaler
import datetime
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from imblearn.over_sampling import SMOTE

sc = StandardScaler()






def scaling_features(x_train, x_test,y_train):
    sc = StandardScaler()

    numerical_training_data = x_train[x_train.select_dtypes(exclude='object').columns]
    categorical_training_data = x_train[x_train.select_dtypes(include='object').columns]

    numerical_test_data = x_test[x_test.select_dtypes(exclude='object').columns]
    categorical_test_data = x_test[x_test.select_dtypes(include='object').columns]

    train_num = sc.fit_transform(numerical_training_data)

    test_num = sc.transform(numerical_test_data)

    k = pd.DataFrame(train_num, index=numerical_training_data.index, columns=numerical_training_data.columns)

    l = pd.DataFrame(test_num, index=numerical_test_data.index, columns=numerical_test_data.columns)

    X_train = pd.concat([k, categorical_training_data], axis=1)
    X_test = pd.concat([l, categorical_test_data], axis=1)

    return X_train,X_test


def feature_selection(x_train,x_test,y_train):
    if  y_train.dtype=='object':
        lb=LabelEncoder()
        y=lb.fit_transform(y_train)
    corr = []
    cor=['RevolvingUtilizationOfUnsecuredLines', 'age', 'MonthlyIncome',
       'DebtRatio', 'NumberOfOpenCreditLinesAndLoans',
       'NumberRealEstateLoansOrLines', 'NumberOfDependents']
    print()
    for i in cor:
        sol = pearsonr(x_train[i], y)
        corr.append(sol)
    corr = np.array(corr)
    p_value_num = pd.Series(corr[: , 1],index=['RevolvingUtilizationOfUnsecuredLines', 'age', 'MonthlyIncome',
       'DebtRatio', 'NumberOfOpenCreditLinesAndLoans',
       'NumberRealEstateLoansOrLines', 'NumberOfDependents'])

    #p_value_num = p_value_num.sort_values(ascending=True)

    print(f"p value num is : {p_value_num}")
    p_value_num.plot.bar()
    plt.show()
    for j in range(len(p_value_num)):
        if p_value_num[j] > 0.05:
            #a = p_value_num[j].index
            x_train = x_train.drop([cor[j]], axis=1)
            x_test = x_test.drop([cor[j]], axis=1)
            print(f"removed {cor[j]} ")

        else:
            print(f"not removed {cor[j]} ")

    return x_train,x_test









def converting_catger_to_num(x_train, x_test,y_train,y_test):

    categorical_training_data = x_train[x_train.select_dtypes(include='object').columns]
    categorical_test_data = x_test[x_test.select_dtypes(include='object').columns]

    if  y_train.dtype=='object':
        lb=LabelEncoder()
        y_train=lb.fit_transform(y_train)
        y_test=lb.transform(y_test)
    for i in x_train.select_dtypes(include='object').columns:
        if len(x_train[i].unique()) <= 5:
            print("before converting")
            x_train[i].unique()
            x_test[i].unique()
            x_train[i]
            x_test[i]
            # for j in range(len(x_train[i].unique())):
            print({x_train[i].unique()[j]: j for j in range(len(x_train[i].unique()))})
            x_test[i] = x_test[i].map({x_test[i].unique()[j]: j for j in range(len(x_test[i].unique()))}).astype(int)
            x_train[i] = x_train[i].map({x_train[i].unique()[j]: j for j in range(len(x_train[i].unique()))}).astype(
                int)

            print("after converting")

            print(f"for {i} uniques are {x_train[i].unique()}")
            print(f"for {i} uniques are {x_test[i].unique()}")


        else:

            print(f"for {i} is not done....")
    return x_train,y_train, x_test,y_test



def balancing_data(x_train,y_train):
    print('The value 1 in dependent variable = ', sum(y_train == 1))
    print('The value 1 in dependent variable = ', sum(y_train == 0))

    # for maintain data balacned we are using upsampling


    sm = SMOTE(random_state=2)

    X_train, Y_train = sm.fit_resample(x_train, y_train)
    print('--------------------------------------------------------------------------')

    print('The value 1 in dependent variable = ', sum(y_train == 1))
    print('The value 1 in dependent variable = ', sum(y_train == 0))

    return X_train,Y_train