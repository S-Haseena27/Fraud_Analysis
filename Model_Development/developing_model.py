from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve


# Logistic Regression
def log_reg(X_train_res, y_train_res, X_test_1, y_test):
    reg1 = LogisticRegression()
    reg1.fit(X_train_res, y_train_res)
    print('Accuracy:')
    print('Training accuracy = ', reg1.score(X_train_res, y_train_res))
    print('Test accuracy = ', reg1.score(X_test_1, y_test))
    y_test_pred = reg1.predict(X_test_1)
    print('Test data confusion_matrix : ', confusion_matrix(y_test, y_test_pred))
    print('Test data classification_report : ', classification_report(y_test, y_test_pred))


# KNN

def knn(X_train_res, y_train_res, X_test_1, y_test):
    reg = KNeighborsClassifier(n_neighbors=5)
    reg.fit(X_train_res, y_train_res)
    print('Accuracy:')
    print('Training accuracy = ', reg.score(X_train_res, y_train_res))
    print('Test accuracy = ', reg.score(X_test_1, y_test))
    y_test_pred = reg.predict(X_test_1)
    print('Test data confusion_matrix : ', confusion_matrix(y_test, y_test_pred))
    print('Test data classification_report : ', classification_report(y_test, y_test_pred))


# Naive bayes
def naive_bayes(X_train_res, y_train_res, X_test_1, y_test):
    reg2 = GaussianNB()
    reg2.fit(X_train_res, y_train_res)
    print('Accuracy:')
    print('Training accuracy = ', reg2.score(X_train_res, y_train_res))
    print('Test accuracy = ', reg2.score(X_test_1, y_test))
    y_test_pred = reg2.predict(X_test_1)
    print('Test data confusion_matrix : ', confusion_matrix(y_test, y_test_pred))
    print('Test data classification_report : ', classification_report(y_test, y_test_pred))


# Decision Tree
def decision_tree(X_train_res , y_train_res,X_test_1 , y_test):
    reg3 = DecisionTreeClassifier(criterion='entropy')
    reg3.fit(X_train_res , y_train_res)
    print('Accuracy:')
    print('Training accuracy = ',reg3.score(X_train_res,y_train_res))
    print('Test accuracy = ',reg3.score(X_test_1,y_test))
    y_test_pred = reg3.predict(X_test_1)
    print('Test data confusion_matrix : ',confusion_matrix(y_test,y_test_pred))
    print('Test data classification_report : ',classification_report(y_test,y_test_pred))


# Random Forest
def random_forest(X_train_res, y_train_res, X_test_1, y_test):
    reg4 = RandomForestClassifier()
    reg4.fit(X_train_res, y_train_res)
    print('Accuracy:')
    print('Training accuracy = ', reg4.score(X_train_res, y_train_res))
    print('Test accuracy = ', reg4.score(X_test_1, y_test))
    y_test_pred = reg4.predict(X_test_1)
    print('Test data confusion_matrix : ', confusion_matrix(y_test, y_test_pred))
    print('Test data classification_report : ', classification_report(y_test, y_test_pred))


def developing_models(X_train_res, y_train, X_test_1, y_test):
    print()
    print("---------------------------------------")
    print('------------*------KNN-----*-----------')
    knn(X_train_res, y_train, X_test_1, y_test)
    print()
    print("---------------------------------------")
    print('------------*------Logistic Regression-----*-----------')
    log_reg(X_train_res, y_train, X_test_1, y_test)
    print()
    print("---------------------------------------")
    print('------------*------Naive Bayes-----*-----------')
    naive_bayes(X_train_res, y_train, X_test_1, y_test)
    print()
    print("---------------------------------------")
    print('------------*------Decision Tree-----*-----------')
    decision_tree(X_train_res, y_train, X_test_1, y_test)
    print()
    print("---------------------------------------")
    print('------------*------Random Forest-----*-----------')
    random_forest(X_train_res, y_train, X_test_1, y_test)


def developing_AUC_ROC_curves(X_train,X_test,y_train,y_test):
    r1 = KNeighborsClassifier()
    r2 = LogisticRegression()
    r3 = GaussianNB()
    r4 = DecisionTreeClassifier()
    r5 = RandomForestClassifier()

    # Model training
    r1.fit(X_train, y_train)
    r2.fit(X_train, y_train)
    r3.fit(X_train, y_train)
    r4.fit(X_train, y_train)
    r5.fit(X_train, y_train)

    # Model testing
    y_K = r1.predict_proba(X_test)[:, 1]
    y_L = r2.predict_proba(X_test)[:, 1]
    y_N = r3.predict_proba(X_test)[:, 1]
    y_D = r4.predict_proba(X_test)[:, 1]
    y_R = r5.predict_proba(X_test)[:, 1]

    # finding FPR and TPR for all models

    fprk, tprk, threshold = roc_curve(y_test, y_K)  # KNN
    fprL, tprL, threshold = roc_curve(y_test, y_L)  # LR
    fprN, tprN, threshold = roc_curve(y_test, y_N)  # NB
    fprD, tprD, threshold = roc_curve(y_test, y_D)  # DT
    fprR, tprR, threshold = roc_curve(y_test, y_R)  # RF

    plt.plot([0, 1], [0, 1], "k--")

    plt.plot(fprk, tprk, label="KNN", color='r')
    plt.plot(fprL, tprL, label="LR", color='g')
    plt.plot(fprN, tprN, label="NB", color='b')
    plt.plot(fprD, tprD, label="DT", color='y')
    plt.plot(fprR, tprR, label="RF", color='black')

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve - for 5 Models")
    plt.legend(loc=0)
    plt.show()
