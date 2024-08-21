#pip install openpyxl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import seaborn as sns

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
df = df.drop(['age_desc', 'used_app_before'], axis = 1) #Drop age_desc and used_app_before column 

#Check if there are any columns with null values; 4 age values were missing
df.isnull().sum()

# change the missing values by the mean value of the attribute
# rounded off to the tenth decimal place
df = df.fillna(round(df.age.mean(), 1))

#Rename some column names
df = df.rename({'jundice': 'jaundice', 'austim': 'autism', 'contry_of_res': 'country_of_res'}, axis = 1)

# Class/ASD is the dependent variable
dv = df.iloc[:,-1].values

# The rest are independent variables
iv = df.iloc[:,0:-1].values

#Make the gender, jaundice, autism, and class/ads columns have values of 0 or 1
lecols = ['gender', 'jaundice', 'autism', 'Class/ASD']
df[lecols] = df[lecols].apply(LabelEncoder().fit_transform)

ohcols = df[['ethnicity', 'country_of_res']]
ohcols = OneHotEncoder().fit_transform(ohcols).toarray()


'''Data Visualization'''
# Histogram for gender attribute
genderHist = df[['gender', 'Class/ASD']]
# gender['gender'].hist(by = genderHist['Class/ASD'], figsize = (5, 2))

# Histogram for ethnicity attribute
ethHist = df[['ethnicity', 'Class/ASD']]
# ethHist['ethnicity'].hist(by = ethHist['Class/ASD'], layout = (1, 2), figsize = (10, 5))

# Boxplot for age and result attributes
age = df[['age', 'result']]
#sns.boxplot(x = 'age', y = 'result', data = age)

