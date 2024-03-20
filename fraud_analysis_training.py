import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from scipy import stats
import logging
from Data_info import read_the_data
from Data_Preprocessing import preprocessing
from Feature_Engineering import feature_engineering
from sklearn.preprocessing import StandardScaler
import datetime
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from Model_Development import developing_model
from sklearn.metrics import roc_auc_score,roc_curve


log_dir=r"D:\ML\Fraud_Analysis"

def configure_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s]- %(message)s',
        filename=os.path.join(log_dir,log_file),
    )


diff_std_orig= None
class Main():
    def __init__(self):
        self.x_train, self.y_train, self.x_test, self.y_test = preprocessing.splitting_data(df)
    def preprocessing_data(self,df):
        #x_train,y_train,x_test,y_test=preprocessing.splitting_data(df)
        self.X_train, self.Y_train, self.X_test, self.Y_test = preprocessing.removing_duplicate_columns(self.x_train,self.x_test,self.y_train,self.y_test)
        self.x_train, self.y_train, self.x_test, self.y_test =preprocessing.handling_null_values_for_dependent_feature(self.X_train,self.Y_train,self.X_test,self.Y_test)
        self.x_train, self.y_train, self.x_test, self.y_test=preprocessing.calling_fun(self.x_train, self.x_test, self.y_train, self.y_test)
        self.x_train, self.y_train, self.x_test, self.y_test=preprocessing.variable_trans_and_outliers_treatment(self.x_train, self.x_test,self.y_train,self.y_test)


        print(f"the values are ........{self.x_train.shape}")
        print(f"the values are ........{self.y_train.shape}")
        print(f"the values are ........{self.x_test.shape}")
        print(f"the values are ........{self.y_test.shape}")

        preprocessing.checking_outliers(self.x_train,self.x_test,self.y_train,self.y_test)


        print(self.x_train.info())
        print(self.x_test.info())


        self.x_train,self.x_test=feature_engineering.scaling_features(self.x_train,self.x_test,self.y_train)

        print(self.x_train.info())
        print(self.x_test.info())

        logging.info(".......After Scaling the values .......")
        logging.info(self.x_train.info())
        logging.info(self.x_test.info())
        logging.info(self.x_train.head())
        logging.info(self.x_test.head())

        print(self.x_train.head())
        print(self.x_test.head())

        self.x_train,self.x_test =feature_engineering.feature_selection(self.x_train,self.x_test,self.y_train)


        self.x_train,self.y_train,self.x_test,self.y_test=feature_engineering.converting_catger_to_num(self.x_train,self.x_test,self.y_train,self.y_test)

        logging.info("......After converting categorical to numerical  .......")

        a=self.x_train.info()
        logging.info(a)

        b=self.x_test.info()
        logging.info(b)

        self.x_train,self.y_train=feature_engineering.balancing_data(self.x_train,self.y_train)


        print(self.x_train.shape)
        print(self.x_test.shape)

        print(self.x_train.info())
        print(self.x_test.info())

        developing_model.developing_models(self.x_train, self.y_train, self.x_test, self.y_test)

        developing_model.developing_AUC_ROC_curves(self.x_train,self.x_test,self.y_train,self.y_test)






if __name__ == '__main__':
    current_date = datetime.datetime.now()
    file_name = f'fraud_analysis_logging.log_{current_date.strftime("%Y-%m-%d-%H-%M-%S")}'
    configure_logging(file_name)
    dataset_path = r"C:\Users\shaik\Downloads\Data Science\ML\ML_END_TO_END_Project\creditcard.csv"

    df = read_the_data.data_information(dataset_path)
    obj=Main()
    obj.preprocessing_data(df)
