# Goodeats
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import plot_tree
from matplotlib import pyplot
from sklearn import tree


import numpy as np
data = pd.read_excel('/Users/akshata/Desktop/food+establishment+violations_Final.xlsx',engine='openpyxl')
#print(data)
print(data['SCORE'].describe())
data['RESULTDTTM'] = pd.to_datetime(data['RESULTDTTM'])
print(data.dtypes)
total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print("missing_data",missing_data)


for column in ['LICSTATUS', 'LICENSECAT' , 'RESULT', 'VIOLSTATUS','GRADE','VIOLATION_ID',]:
    data[column].fillna(data[column].mode()[0], inplace=True)

# # Define a mapping from string labels to integer values
# violation_mapping = {'Fail': 0, 'Pass': 1, }

# # Encode violation status as integers using replace()
# data['VIOLSTATUS'] = data['VIOLSTATUS'].replace(violation_mapping)

# Print the updated DataFrame

filtered_df1 = data[(data['VIOLSTATUS'] == "Fail")]
filtered_df2 = data[(data['VIOLSTATUS'] == "Pass")]

filtered_df = pd.concat([filtered_df1, filtered_df2], axis=0)
print("filter_df",filtered_df)
print("filter_df",filtered_df2['VIOLSTATUS'])
data=data.drop(data.index[324641:])

data = data.replace(to_replace="FT",value=0)
data = data.replace(to_replace="FS",value=1)
data = data.replace(to_replace="RF",value=2)
data = data.replace(to_replace="MFW",value=3)

data = data.replace(to_replace="Active",value=1)
data = data.replace(to_replace="Inactive",value=2)
data = data.replace(to_replace="Deleted",value=0)

data = data.replace(to_replace="Fail",value=0)

data = data.replace(to_replace="Pass",value=1)
print(data['VIOLSTATUS'])
data = data.replace(to_replace="A",value=1)
data =data.replace(to_replace="B",value=2)
data =data.replace(to_replace="C",value=3)

total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

for column in ['LICSTATUS', 'LICENSECAT' , 'RESULT', 'VIOLSTATUS','GRADE','VIOLATION_ID','RESULTDTTM',]:
    data[column].fillna(data[column].mode()[0], inplace=True)

data = data.replace(to_replace="FT",value=0)
data = data.replace(to_replace="FS",value=1)
data = data.replace(to_replace="RF",value=2)
data = data.replace(to_replace="MFW",value=3)

data = data.replace(to_replace="Active",value=1)
data = data.replace(to_replace="Inactive",value=2)
data = data.replace(to_replace="Deleted",value=0)

data = data.replace(to_replace="Fail",value=0)

data = data.replace(to_replace="Pass",value=1)

data = data.replace(to_replace="A",value=1)
data =data.replace(to_replace="B",value=2)
data =data.replace(to_replace="C",value=3)
data=data.replace({'RESULT': {'HE_Pass' :1, 'HE_Fail' :2,'HE_Filed':3,'HE_Closure' :4,'HE_FailExt' :5,'HE_FAILNOR' :6,'HE_Hearing' :7,'HE_Misc' :8,'HE_NotReq' :9,'HE_OutBus' :10,'HE_TSOP' :11,'Pass' :12}})
print(data['VIOLSTATUS'])
################ replacing Grade misssing with mean value ###########
mean_value_grade = data['GRADE'].mean()
data['GRADE'].fillna(value=mean_value_grade, inplace=True)

################ replacing Score misssing with mean value ###########
mean_value_score = data['SCORE'].mean()
data['SCORE'].fillna(value=mean_value_score, inplace=True)
# print(data['SCORE'].head(20))
# print('Updated Dataframe:')
# print(data)
################ replacing SUM_VIOLATIONS missing value with mean value ###########
mean_value_sum_violations = data['SUM_VIOLATIONS'].mean()
data['SUM_VIOLATIONS'].fillna(value=mean_value_sum_violations, inplace=True)
mean_value_LICENSECAT = data['LICENSECAT'].mean()
data['LICENSECAT'].fillna(value=mean_value_LICENSECAT, inplace=True)
# print('Updated Dataframe:')

feature_cols = ['GRADE','RESULT','SUM_VIOLATIONS','LICENSECAT','LICENSENO','LICSTATUS',]
X = data[feature_cols] # Features
y = filtered_df['VIOLSTATUS'] # Target variable

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)
print(X_train)
print(y_train)
# print(type(y_train))



label_encoder = preprocessing.LabelEncoder()
# y_train = label_encoder.fit_transform(y_train)
# X_train = label_encoder.fit_transform(X_train)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_leaf_nodes= 5,random_state=0)
clf = clf.fit(X_train,y_train)

 # Train Decision Tree Classifer
#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy_Decision Tree:",metrics.accuracy_score(y_test, y_pred))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Accuracy_Random Forest:",metrics.accuracy_score(y_test, y_pred))
logreg= LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print (X_test) #test dataset
print (y_pred) #predicted values

print("Accuracy_LogReg:",metrics.accuracy_score(y_test, y_pred))
text_representation = tree.export_text(clf)
print(text_representation)

plt.figure()
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
# plot_tree(clf, filled=True)
# plt.title("Decision tree trained on all the features")
# plt.show()

# get importance
importance = clf.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
