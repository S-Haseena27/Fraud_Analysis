import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Data_info import read_the_data
from sklearn.model_selection import train_test_split
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import math

sc = StandardScaler()
diff_std_orig= []

standard_deviations={}
tech=['mean','mode','median','random_sample']



def splitting_data(df):

    X = df.iloc[:, :-1]  # independent variables
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)
    logging.info(f"The shape of x_train is {x_train.shape} ")
    logging.info(f"The shape of y_train is {y_train.shape}" )
    logging.info(f"The shape of x_test is {x_test.shape} " )
    logging.info(f"The shape of y_test is {y_test.shape} " )
    return x_train,y_train,x_test,y_test


def removing_duplicate_columns(x_train,x_test,y_train,y_test):
    #  Removing Duplicate Columns
    std = []
    rem = []
    for i in x_train.select_dtypes(exclude='object').columns:
        std.append(x_train[i].std())
    for i in range(len(std)):
        for j in range(i):
            if std[i] == std[j]:
                rem.extend([[i, j]])
    if len(rem) != 0:
        for i in rem:
            x_train = x_train.drop([x_train.select_dtypes(exclude='object').columns[i[0]]], axis=1)
            x_test = x_test.drop([x_test.select_dtypes(exclude='object').columns[i[0]]], axis=1)

    x_test.info()
    x_train.info()
    #handling_null_values_for_dependent_feature(x_train, y_train, x_test, y_test)
    return x_train,y_train,x_test,y_test

def handling_null_values_for_dependent_feature(x_train,y_train,x_test,y_test):
    print('treating null values')
    x_train.info()
     #try:

         ##### Doubt
     # except Exception as e:
     #     logging.error(e)

     #c=y_train.isnull().sum()

    logging.info(f" null values in y_train are {y_train.isnull().sum()}")
     #d=y_test.isnull().sum()
    logging.info(f" null values in y_test are {y_test.isnull().sum()}")
     # print(c)
     # print(d)

     # Handling null values for dependent data
     #try:
    if ((y_train.isnull().sum()) > 0):
        training_data=pd.concat([x_train,y_train],axis=1)
        ind=training_data[training_data.iloc[:,-1].isnull()].index.tolist()
        training_data=training_data.drop(ind,axis=0)
        x_train=training_data.iloc[:,:-1]
        y_train=training_data.iloc[:,-1]
        logging.info(f" null values in y_train are {y_train.isnull().sum()}")

    elif ((y_test.isnull().sum()) > 0):
        testing_data=pd.concat([x_test,y_test],axis=1)
        ind = testing_data[testing_data.iloc[:, -1].isnull()].index.tolist()
        testing_data = testing_data.drop(ind, axis=0)
        x_test=testing_data.iloc[:,:-1]
        y_test=testing_data.iloc[:,-1]
     # except Exception as e:
     #     logging.error(e)
    x_train['NumberOfDependents'] = pd.to_numeric(x_train['NumberOfDependents'])
    x_test['NumberOfDependents'] = pd.to_numeric(x_test['NumberOfDependents'])
    print("after treating null values...........")
    x_train.info()
    #calling_fun(x_train, x_test, y_train, y_test)
    return x_train,y_train,x_test,y_test






def hand_null_for_independent_feature(x_train,x_test,i,y_train,y_test):
    # # Handling null values for independent data
    #
    # a = x_train.isnull().sum()
    # logging.info(f" null values in x_train are {a}")
    # b = x_test.isnull().sum()
    # logging.info(f" null values in x_test are {b}")


    if x_train[i].isnull().sum() !=0:
        mean_replacement(x_train, i)
        mode_replacement(x_train, i)
        median_replacement(x_train, i)
        random_sample(x_train, i)
        apply_best_technique(x_train,x_test, i)
        print(x_train.info())
        EDA(x_train,i)
    #
    #  "call functions"
    #
    #
    #      calling_fun(x_train, x_test, i)
    #
     ## For categorical features
    #try:
    for j in x_train.select_dtypes(include='object').columns:
        m=x_train[j].mode()
        if x_train[j].isnull().sum() !=0:
            x_train[j]=x_train[j].fillna(m)
            x_test[j]=x_test[j].fillna(m)

    return x_train,y_train,x_test,y_test
    # except Exception as E:
    #     logging.error(E)

    #

def calling_fun(x_train,x_test,y_train,y_test):
    # hand_null_for_independent_feature(x_train, x_test, i)

    for i in x_train.columns:
        x_train,y_train,x_test,y_test=hand_null_for_independent_feature(x_train, x_test, i,y_train,y_test)

    x_train.info()
    x_test.info()
    for i in x_train.columns:
        x_train=removing_replaced_columns(x_train,x_test,i,y_train,y_test)
        print("============================================")
        print("============================================")
        logging.info(f"removing replaced columns are {x_train.info()}")
        print("============================================")
        print("============================================")

        #x_train=removing_replaced_columns(x_train,i)

    checking_normal_dist(x_train, x_test,y_train,y_test)
    #variable_trans_and_outliers_treatment(x_train, x_test,y_train,y_test)
    return x_train,y_train,x_test,y_test





def mean_replacement(dataset, var):
    global diff_std_orig
    diff_std_orig = []
    diff_std = []
    mean = dataset[var].mean()
    std = dataset[var].std()
    dataset[var + "_mean"] = dataset[var].fillna(mean)
    std_mean = dataset[var + "_mean"].std()
    standard_deviations['mean_std'] = std_mean
    #try:
    diff_std.append(abs(std - std_mean))
    diff_std_orig.append(diff_std[0])
    #except Exception as e:
        #logging.error(e)
    logging.info(f"std of replaced mean is {std_mean} ")
    logging.info(f"std of original feature is {std} \n")
    logging.info("====================================== \n")


def mode_replacement(dataset, var):
    global diff_std_orig
    mode = dataset[var].mode()[0]
    std = dataset[var].std()
    dataset[var + "_mode"] = dataset[var].fillna(mode)
    std_mode = dataset[var + "_mode"].std()
    standard_deviations['mode_std'] = std_mode

    #try:
    diff_std_orig.append(abs(std - std_mode))
    # except Exception as e:
    #     logging.error(e)

    logging.info(f"std of replaced mode is {std_mode} ")
    logging.info(f"std of original feature is {std} \n ")
    logging.info("====================================== \n")


def median_replacement(dataset, var):
    global diff_std_orig
    median = dataset[var].median()
    std = dataset[var].std()
    dataset[var + "_median"] = dataset[var].fillna(median)
    std_median = dataset[var + "_median"].std()
    standard_deviations['median_std'] = std_median

    # try:
    diff_std_orig.append(abs(std - std_median))
    # except Exception as e:
    #     logging.error(e)

    logging.info(f"std of replaced median is {std_median} ")
    logging.info(f"std of original feature is {std} \n")
    logging.info("======================================")


def random_sample(dataset, var):
    global diff_std_orig
    std = dataset[var].std()
    dataset[var + "_replaced"] = dataset[var].copy()
    s = dataset[var].dropna().sample(dataset[var].isnull().sum(), random_state=42)
    s.index = dataset[dataset[var].isnull()].index
    dataset.loc[dataset[var].isnull(), var + "_replaced"] = s

    std_sample = dataset[var + "_replaced"].std()
    standard_deviations['sample_replaced'] = std_sample

    # try:
    diff_std_orig.append(abs(std - std_sample))
    # except Exception as e:
    #     logging.error(e)

    logging.info(f"std of sample technique is {std_sample} ")
    logging.info(f"std of original feature is {std} \n")
    logging.info("======================================")
    #apply_best_technique(dataset, var)


def apply_best_technique(x_train, x_test,var):
    global diff_std_orig
    mean = x_train[var].mean()
    mode = x_train[var].mode()[0]
    median = x_train[var].median()
    ind = diff_std_orig.index(min(diff_std_orig))

    '''
    # 0 th index is  mean 
    # 1 st index is mode and 
    # 2nd index is median
    # 3 rd index is random sample
    '''
    #try:
    if ind == 0:
        x_train[var] = x_train[var].fillna(mean)
        x_test[var]=x_test[var].fillna(mean)
    elif ind == 1:
        x_train[var] = x_train[var].fillna(mode)
        x_test[var] = x_test[var].fillna(mode)
    elif ind == 2:
        x_train[var] = x_train[var].fillna(median)
        x_test[var] = x_test[var].fillna(median)
    else:
        s = x_train[var].dropna().sample(x_train[var].isnull().sum(), random_state=42)
        s.index = x_train[x_train[var].isnull()].index
        x_train.loc[x_train[var].isnull(), var] = s
        # doubt
        t=x_test[var].dropna().sample(x_test[var].isnull().sum(),random_state=42)
        t.index=x_test[x_test[var].isnull()].index
        x_test.loc[x_test[var].isnull(), var] = t
    # except Exception as e:
    #     logging.error(e)
    logging.info(x_train[var].std())
    logging.info(f"{tech[ind]} technique is used for null values treatmet in {var} feature.....")



def EDA(X_train,var):
    #try:
    X_train[var].plot(kind = 'kde',color = 'r' , label = var+'_original')
    X_train[var+'_mean'].plot(kind = 'kde',color = 'g' , label = var+'_mean')
    X_train[var+'_median'].plot(kind = 'kde',color = 'b' , label = var+'_median')
    X_train[var+'_mode'].plot(kind = 'kde',color = 'y' , label = var+'_mode')
    X_train[var+'_replaced'].plot(kind = 'kde',color = 'black' , label = var+'_randomsample')
    plt.legend()
    plt.show()

    # except Exception as err:
    #     logging.error(err)


def removing_replaced_columns(X_train,x_test,var,y_train,y_test):
    #try:
    if var[-4:] == 'mode' or var[-4:] == 'mean' or var[-6:] == 'median' or var[-8:] == 'replaced':
        X_train = X_train.drop(var, axis=1)
        logging.info(f"{var} is deleted from dataset.......")
        return X_train
    else:
        logging.info(f"{var} is not deleted from dataset.......")
        return X_train
    # except Exception as err:
    #     logging.error(err)

    #checking_normal_dist(X_train,x_test)

def checking_normal_dist(x_train,x_test,y_train,y_test):
    a=len(x_train.select_dtypes(exclude='object').columns)
    j=0
    plt.figure(figsize=(20,22))
    for i in x_train.select_dtypes(exclude='object').columns:
        j+=1
        plt.subplot(int(a/2)+1,2,j)
        sns.distplot(x_train[i])
    plt.show()

    checking_outliers(x_train,x_test,y_train,y_test)



def checking_outliers(x_train,x_test,y_train,y_test):
    #try:
    a=len(x_train.select_dtypes(exclude='object').columns)
    j=0
    plt.figure(figsize=(20,15))
    for i in x_train.select_dtypes(exclude='object').columns:
        j+=1
        plt.subplot(int(a/2)+1,2,j)
        x_train[i].plot(kind='box')
    plt.show()
    # except Exception as e:
    #     logging.error(e)




def variable_trans_and_outliers_treatment(x_train,x_test,y_train,y_test):

    """Before yeojhonsan technique......"""
    print("Before yeojhonsan technique")
    for k in x_train.select_dtypes(exclude='object').columns:
        variable_transformation(x_train, k)

    """after yeojhonsan technique......"""

    print("After yeojhonsan technique")

    for l in x_train.select_dtypes(exclude='object').columns:
        x_train[l], alpha = stats.yeojohnson(x_train[l])
        x_test[l],beta=stats.yeojohnson(x_test[l])
        variable_transformation(x_train, l )
        print()
        print("For Test data")
        variable_transformation(x_test, l)



    ### Applying IQR technique

    for i in x_train.select_dtypes(exclude='object').columns:
        upper_lim, lower_lim = iqr_tech(x_train, i)
        x_train[i] = np.where(x_train[i] > upper_lim, upper_lim,
                              np.where(x_train[i] < lower_lim, lower_lim,x_train[i]))
        print(f"for column {i} lower is {lower_lim},upper is {upper_lim}")
        if lower_lim==0.0 or upper_lim ==0.0:
            print(f"deleting {i}")
            x_train=x_train.drop([i],axis=1)
            x_test=x_test.drop([i],axis=1)
    print(x_train.info())

    checking_outliers(x_train, x_test,y_train,y_test)


    """For test Data"""

    for i in x_test.select_dtypes(exclude='object').columns:
        upper_lim, lower_lim = iqr_tech(x_test, i)
        x_test[i] = np.where(x_test[i] > upper_lim, upper_lim,
                              np.where(x_test[i] < lower_lim, lower_lim,x_test[i]))
        print(f"lower is {lower_lim},upper is {upper_lim}")

    checking_outliers(x_test,x_train,y_train,y_test)
    X_train=x_train
    print("x_train is assigned")
    X_test=x_test
    print("x_test is assigned")
    Y_train=y_train
    print("y_train is assigned")
    Y_test=y_test
    print("y_test is assigned")


    print(f"The shape is {X_train.shape}")
    print(f"The shape is {Y_train.shape}")
    print(f"The shape is {X_test.shape}")
    print(f"The shape is {Y_test.shape}")



    return X_train,Y_train,X_test,Y_test


def variable_transformation(x_train, i):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('boxplot')
    plt.xlabel(f"{i}")
    plt.boxplot(x_train[i])

    plt.subplot(1, 3, 2)
    plt.title('hist')
    plt.xlabel(f"{i}")
    plt.hist(x_train[i])

    plt.subplot(1, 3, 3)
    plt.title('probability_plot')
    plt.xlabel(f" {i} ")
    stats.probplot(x_train[i], dist='norm', plot=plt)

def iqr_tech(dataset,var):
    iqr_value = dataset[var].quantile(0.75)-dataset[var].quantile(0.25)
    upper_lim=dataset[var].quantile(0.75) + (1.5*iqr_value)
    lower_lim=dataset[var].quantile(0.25) - (1.5*iqr_value)
    return upper_lim,lower_lim


