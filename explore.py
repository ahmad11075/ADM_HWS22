import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score,f1_score, roc_curve, accuracy_score


# Preprocessing the dataset
def preprocessing(data):
    # replace with numerical values
    data['Dependents'].replace('3+', 3, inplace=True)
    data['Loan_Status'].replace('N', 0, inplace=True)
    data['Loan_Status'].replace('Y', 1, inplace=True)
    data['Gender'].replace('Male', 1, inplace=True)
    data['Married'].replace('Yes', 1, inplace=True)
    data['Education'].replace('Graduate', 1, inplace=True)
    data['Self_Employed'].replace('Yes', 1, inplace=True)
    data['Gender'].replace('Female', 0, inplace=True)
    data['Married'].replace('No', 0, inplace=True)
    data['Education'].replace('Not Graduate', 0, inplace=True)
    data['Self_Employed'].replace('No', 0, inplace=True)
    # Handle missing data
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['Married'].fillna(data['Married'].mode()[0], inplace=True)
    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
    data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)
    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
    data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
    # Drop ID column
    data = data.drop('Loan_ID', axis=1)

    # Split features and target
    X = data.drop('Loan_Status', axis=1)
    y = data.Loan_Status.values

    return X, y
# Read data using pandas
# pd.read_csv returns pandas dataframe
data = pd.read_csv('loan_data_set.csv')

# Precprocess data
X, y = preprocessing(data)

# Show the new distribution of classes after resampling
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# Lists of comparison index
modelName = list()
precision = list()
recall = list()
F1score = list()
AUCROC = list()
CM = list()
accuracy = list()



def test_eval(clf_model, X_test, y_test, algo=None):
    # Test set prediction
    y_pred=clf_model.predict(X_test)   
    modelName.append(algo)
    precision.append(precision_score(y_test,y_pred))
    recall.append(recall_score(y_test,y_pred))
    F1score.append(f1_score(y_test,y_pred))
    AUCROC.append(roc_auc_score(y_test, y_pred))
    CM.append(confusion_matrix(y_pred, y_test, labels=[0,1]))
    accuracy.append(accuracy_score(y_test, y_pred))

# Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Import Descision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = "entropy")
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

# Show DecisionTree scores
test_eval(model, X_test, y_test, 'DecisionTree')
comparison = pd.DataFrame({'Model':modelName,
                            'Precision':precision,
                            'Recall':recall,
                            'F1-Score':F1score,
                            'AUC-ROC':AUCROC,
                            'Accuracy':accuracy})


# Import Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
LRmodel = LogisticRegression()
LRmodel.fit(X_train, y_train)
y_pred=LRmodel.predict(X_test)

# Show Logistic Regression scores
test_eval(LRmodel, X_test, y_test, 'Logistic Regression')
comparison = pd.DataFrame({'Model':modelName,
                            'Precision':precision,
                            'Recall':recall,
                            'F1-Score':F1score,
                            'AUC-ROC':AUCROC,
                            'Accuracy':accuracy})
# Import GaussianNB Classifier
from sklearn.naive_bayes import GaussianNB
NBmodel = GaussianNB()
NBmodel.fit(X_train, y_train)
y_pred=NBmodel.predict(X_test)

# Show Naive Bayes scores
test_eval(NBmodel, X_test, y_test, 'Naive Bayes')
comparison = pd.DataFrame({'Model':modelName,
                            'Precision':precision,
                            'Recall':recall,
                            'F1-Score':F1score,
                            'AUC-ROC':AUCROC,
                            'Accuracy':accuracy})


# Import KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
KNNmodel = KNeighborsClassifier( n_neighbors= 3 )
KNNmodel.fit(X_train, y_train)
y_pred=KNNmodel.predict(X_test)


# Show KNN scores
test_eval(KNNmodel, X_test, y_test, 'KNN')
comparison = pd.DataFrame({'Model':modelName,
                            'Precision':precision,
                            'Recall':recall,
                            'F1-Score':F1score,
                            'AUC-ROC':AUCROC,
                            'Accuracy':accuracy})



# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Scalling the input data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Import SVC Classifier
from sklearn.svm import SVC
SVMmodel = SVC(kernel='linear')
SVMmodel.fit(X_train, y_train)
y_pred=SVMmodel.predict(X_test)

# Show SVM scores
test_eval(SVMmodel, X_test, y_test, 'SVM')
comparison = pd.DataFrame({'Model':modelName,
                            'Precision':precision,
                            'Recall':recall,
                            'F1-Score':F1score,
                            'AUC-ROC':AUCROC,
                            'Accuracy':accuracy})


def show_explore_page():
    st.title("Comparison of classification classifiers")
    st.table(comparison)


