'''SUPPORT VECTOR MACHINES'''

#pip install openpyxl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


'''Reading the Excel file and converting it to a CSV file'''
#Read the Excel file using pandas
read_file = pd.read_excel('Child Autism Study.xlsx', engine = 'openpyxl')

#Convert to CSV
read_file.to_csv("Test.csv", index = None, header=True)

#Read csv file and put it in a DataFrame object
df = pd.DataFrame(pd.read_csv("Test.csv"))



'''Data Cleaning'''
df = df.drop(df[df.ethnicity == '?'].index) #Drop rows with ethnicity == ?
df = df.drop(df.iloc[:, :10], axis = 1) #Drop the rows and columns for the questionnaire
df = df.drop(['age_desc', 'used_app_before'], axis = 1) #Drop age_desc and used_app_before columns 

#Check if there are any columns with null values; 1 age value was missing
df.isnull().sum()

#Rename some column names
df = df.rename({'jundice': 'jaundice', 'austim': 'autism', 'contry_of_res': 'country_of_res'}, axis = 1)

# Class/ASD is the dependent variable
dv = df.iloc[:,-1].values

# The rest are independent variables
iv = df.iloc[:,0:-1].values

#Make the gender, jaundice, autism, and class/ads columns have values of 0 or 1
#Gender: 0 = female, 1 = male
#Jaundice: 0 = no, 1 = yes
#Autism: 0 = no, 1 = yes
#Class/ASD: 0 = no, 1 = yes
lecols = ['gender', 'jaundice', 'autism', 'Class/ASD']
df[lecols] = df[lecols].apply(LabelEncoder().fit_transform)

# Class/ASD is the dependent variable
dv = df.iloc[:,-1].values

# change the missing values by the mean value of the attribute
# rounded off to the nearest integer
df.loc[df.age == '?'] = '0'
#print(df.age.values)

#change the age column into an integer data type
df['age'] = df['age'].astype(int)
df.loc[df['age'] == 0] = round(df.age.mean())

#Change the other columns into integer type
df['gender'] = df['gender'].astype(int)
df['jaundice'] = df['jaundice'].astype(int)
df['autism'] = df['autism'].astype(int)
df['result'] = df['result'].astype(int)
df['Class/ASD'] = df['Class/ASD'].astype(int)

# The rest are independent variables
features = ['age', 'gender', 'jaundice', 'autism', 'result']
iv = df.loc[:, features].values



'''SVM ALgorithm (LINEAR)'''
#80% training and 20% test
data = df.drop(['ethnicity', 'country_of_res', 'relation'], axis = 1)
target = df['Class/ASD']

#Split into training and test data
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

#Generate the model
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy: how often is the classifier correct?
print("LINEAR KERNEL")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



'''SVM Algorithm (POLYNOMIAL)'''
#Create a svm Classifier
clf = svm.SVC(kernel='poly') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy: how often is the classifier correct?
print("POLYNOMIAL KERNEL")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



'''SVM Algorithm (RBF)'''
#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy: how often is the classifier correct?
print("RBF KERNEL")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))