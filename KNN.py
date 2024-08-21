#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[2]:

#Read the Excel file using pandas
read_file = pd.read_excel('Child Autism Study.xlsx', engine = 'openpyxl')

#Convert to CSV
read_file.to_csv("Test.csv", index = None, header=True)

#Read csv file and put it in a DataFrame object
df = pd.DataFrame(pd.read_csv("Test.csv"))


# In[4]:


dcopy = df
# Data Cleaning
df = df.drop(df[df.ethnicity == "b'?'"].index)
df = df.drop(df.iloc[:, :10], axis = 1)
df = df.drop(['age_desc', 'used_app_before'], axis = 1)


# In[5]:


# find missing values, 4 age values were missing
df.isnull().sum() 


# In[6]:


# change the missing values by the mean value of the attribute
# rounded off to the tenth decimal place
df = df.fillna(round(df.age.mean(), 1))

df = df.rename({'jundice': 'jaundice', 'austim': 'autism', 'contry_of_res': 'country_of_res'}, axis = 1)


# In[14]:


# 
lecols = ['gender', 'jaundice', 'autism', 'Class/ASD', 'ethnicity']
df[lecols] = df[lecols].apply(LabelEncoder().fit_transform)

ohcols = df[['ethnicity', 'country_of_res']]
ohcols = OneHotEncoder().fit_transform(ohcols).toarray()
ohcols


# In[15]:


# Splitting the data set 
x = df[['age', 'gender', 'jaundice', 'autism']]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)


# In[16]:


# Define the model
classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')


# In[17]:


#Fit Model
classifier.fit(x_train, y_train)


# In[19]:


# Predict test results
y_pred = classifier.predict(x_test)
y_pred


# In[21]:


# Evaluate Model
# Print f1 score and accuracy score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test,y_pred))


# In[ ]:




