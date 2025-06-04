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
    
    
    