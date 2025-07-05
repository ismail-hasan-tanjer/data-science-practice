#this data science practice file 

#Data Cleaning & Preprocessing

#Data Cleaning & Preprocessing is one of the most important and initial steps of any data science project. The main objective of this step is to prepare the raw data for analysis or modeling. The main steps are given below in detail:

'''1. Handling Missing Values
Sometimes some fields in the dataset are empty (null/NaN). These need to be managed:

Drop missing values: df.dropna()

Fill with mean/median/mode: df.fillna(df['column'].mean())

Forward/Backward Fill: df.fillna(method='ffill') or bfill'''


'''
2. Removing Duplicates
Having the same information multiple times clutters the data: '''

#code 

df.drop_duplicates(inplace=True)

'''
3. Cleaning Text Data
To clean text or categorical data:

Case conversion: .lower(), .upper()

Removing special characters: Using RegEx

Removing white spaces: .strip()

Tokenization / Lemmatization (in NLP projects)'''



4. Converting Data Types
Using the correct type increases performance and accuracy:
    
#code 
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')



5. Encoding Categorical Variables

Machine learning models only understand numerical input. So:
    
    
    
Label Encoding: eg Male = 0, Female = 1

One-Hot Encoding: pd.get_dummies(df['column'])
    

6. Feature Scaling

The model is confused if the features are too small or too large compared to each other. So:

Standardization: (Z-score)

from sklearn.preprocessing import StandardScaler

Normalization: Scaling between 0 and 1

from sklearn.preprocessing import MinMaxScaler


7. Outlier Detection & Removal
Outliers or exceptional values ​​can reduce the accuracy of the model:
    
    
Z-score method

IQR method

Boxplot visualization


8. Data Splitting


Data is split to train the model:
    
#code    
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#*********************Very Important"

Exploratory Data Analysis (EDA)

This is an important step that helps in understanding the insights and structure within the data.
The detailed explanation and steps of EDA are given below:
    
Exploratory Data Analysis (EDA)
EDA is a process where data visualization, statistics, and various analysis methods are used to gain a deeper understanding of a dataset. In this step, you will understand:

Which features are important

Which features are related

What kind of outliers or distributions are there

How the model will perform can be predicted.


1. Basic Information Check

df.info()
df.describe()
df.head()
df.tail()

=> You can learn about data type, min, max, standard deviation, missing values, etc.


2. Univariate Analysis
To look at the distribution of a single variable:
 
 #Numerical Data:
#code   
import seaborn as sns
sns.histplot(df['age'])
sns.boxplot(x=df['salary'])

#Categorical Data:
df['gender'].value_counts().plot(kind='bar')


3. Bivariate/Multivariate Analysis
To see the relationship between two or more variables:


Scatter Plot (for two numeric variables):

sns.scatterplot(x='age', y='salary', data=df)

Heatmap (Correlation Matrix):
    
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

Pairplot (Multiple feature relationships):
    
sns.pairplot(df)


#4. Correlation Analysis

#To understand the relationship between data features:

#code-

df.corr()

#➡️ This shows which features are strongly correlated (positive/negative) with each other.


5. Target Variable Analysis
If supervised learning (such as classification or regression), then analyze the target variable:

sns.countplot(x='target', data=df)


6. Grouping & Aggregation
Extract insights by grouping by a categorical feature:

df.groupby('gender')['income'].mean()








   
    





    


    
    
    
    