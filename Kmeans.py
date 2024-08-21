#!/usr/bin/env python
# coding: utf-8

# In[491]:


from scipy.io import arff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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

ohcols = df[['ethnicity', 'country_of_res']]
ohcols = OneHotEncoder().fit_transform(ohcols).toarray()


# In[535]:


# Unsupervised (K-Means Algorithm)

x = df[['ethnicity', 'Class/ASD']]
y = df[['jaundice', 'Class/ASD']]

# WCSS and Elbow method to find the good number of clusters to be used
# wcss=[]
# for i in range(1,7):
#     kmeans = KMeans(i)
#     kmeans.fit(x)
#     wcss_iter = kmeans.inertia_
#     wcss.append(wcss_iter)

# number_clusters = range(1,7)
# plt.plot(number_clusters, wcss)
# plt.title('Finding good no. of clusters')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')


# K Means model for ethnicity and Class/ASD
plt.scatter(df['ethnicity'], df['Class/ASD'])
kmeansx = KMeans(3)
kmeansx.fit(x)

clustersx = kmeansx.fit_predict(x)

dcopyx = df.copy()
dcopyx['Clusters'] = clustersx
plt.scatter(dcopyx['ethnicity'],dcopyx['Class/ASD'],c=dcopyx['Clusters'],cmap='rainbow')
plt.xlabel('Ethnicity')
plt.ylabel('Class/ASD')
plt.title('K Means Model for Ethnicity and Class/ASD')
plt.show()

# K Means model for jaundice and Class/ASD
# plt.scatter(df['jaundice'], df['Class/ASD'])
# kmeansy = KMeans(4)
# kmeansy.fit(y)

# clustersy = kmeansy.fit_predict(y)

# dcopyy = df.copy()
# dcopyy['Clusters'] = clustersy
# pltx = plt.scatter(dcopyy['jaundice'],dcopyy['Class/ASD'],c=dcopyy['Clusters'],cmap='rainbow')
# plt.xlabel('Jaundice')
# plt.ylabel('Class/ASD')
# plt.title('K Means Model for Jaundice and Class/ASD')


# In[405]:


# Exploratory Data Analysis

# Histogram for gender attribute
genderHist = df[['gender', 'Class/ASD']]
# genderHist['gender'].hist(by = genderHist['Class/ASD'], figsize = (5, 2))

# Histogram for ethnicity attribute
ethHist = df[['ethnicity', 'Class/ASD']]
# ethHist['ethnicity'].hist(by = ethHist['Class/ASD'], layout = (1, 2), figsize = (10, 5))

# Boxplot for age and result attributes
age = df[['age', 'result']]
# sns.boxplot(x = 'age', y = 'result', data = age)

