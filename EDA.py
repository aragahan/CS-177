'''EXPLORATORY DATA ANALYSIS'''

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):
    if x == None:
        column_interested = y
    else:
        column_interested = x
    series = dataframe[column_interested]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
    plt.show()


def quantitative_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', ax=None, verbose=True, swarm=False):
    series = dataframe[y]
    print(series.describe())
    print('mode: ', series.mode())
    if verbose:
        print('='*80)
        print(series.value_counts())

    sns.boxplot(x=x, y=y, hue=hue, data=dataframe, palette=palette, ax=ax)

    if swarm:
        sns.swarmplot(x=x, y=y, hue=hue, data=dataframe,
                      palette=palette, ax=ax)

    plt.show()

'''Reading the Excel file and converting it to a CSV file'''
#Read the Excel file using pandas
read_file = pd.read_excel('Child Autism Study.xlsx', engine = 'openpyxl')

#Convert to CSV
read_file.to_csv("Test.csv", index = None, header=True)

#Read csv file and put it in a DataFrame object
df = pd.DataFrame(pd.read_csv("Test.csv"))

'''Data Cleaning/Preproccessing'''
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

#CHange the data types to string
df['ethnicity'] = df['ethnicity'].astype("string")
df['country_of_res'] = df['country_of_res'].astype("string")
df['relation'] = df['relation'].astype("string")

#Check what the data types are in the dataframe
print(df.dtypes)



'''EDA Proper'''
#ethnicity
categorical_summarized(df, y='ethnicity')

#country of res
categorical_summarized(df, y='country_of_res')

#relation
categorical_summarized(df, y='relation')

#gender
categorical_summarized(df, y='gender')


#jaundice
categorical_summarized(df, y='jaundice')

#autism
categorical_summarized(df, y='autism')


#age
quantitative_summarized(df, y='age')

#result
quantitative_summarized(df, y='result')


#Correlation
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', size = 15)
colormap = sns.diverging_palette(10, 220, as_cmap = True)
sns.heatmap(df.corr(),
            cmap = colormap,
            square = True,
            annot = True,
            linewidths=0.1,vmax=1.0, linecolor='white',
            annot_kws={'fontsize':12 })
plt.show()