'''PRINCIPAL COMPONENT ANALYSIS'''

#pip install openpyxl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
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


#Check what the data types are in the dataframe
#print(df.dtypes)


#Check the Dataset
#print(df.head())


'''PCA ALgorithm'''
# Class/ASD is the dependent variable
dv = df.iloc[:,-1].values

# change the missing values by the mean value of the attribute
# rounded off to the nearest integer
df.loc[df.age == '?'] = '0'
#print(df.age.values)

#change the age column into an integer data type
df['age'] = df['age'].astype(int)
df.loc[df['age'] == 0] = round(df.age.mean())

# The rest are independent variables
features = ['age', 'gender', 'jaundice', 'autism', 'result']
iv = df.loc[:, features].values

#Perform PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(iv)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Class/ASD']]], axis = 1)

#Variability = 92%
print(pca.explained_variance_ratio_)
print(pca.components_)


#Data Visulaization
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Autism Dataset",fontsize=20)
targets = [0, 1]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep =finalDf['Class/ASD'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.show()


