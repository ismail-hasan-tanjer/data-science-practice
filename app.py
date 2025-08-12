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



7.Outliers & Anomalies Detection
Determine outliers using boxplot or Z-score.


✅ Summary
The purpose of EDA is to:

Understand the nature and structure of data

Identify important patterns, trends, and relationships

Create decision support through data visualization

Assist in feature engineering by understanding which features need to be discarded or retained


#Matplotlib

📊 Data Visualization using Matplotlib
Matplotlib is the most popular and basic data visualization library in Python. With its help, you can create line plots, bar charts, histograms, scatter plots, etc.


🔧 1. Installation (if not installed yet)

pip install matplotlib
    

🧱 2. Basic Structure


import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

plt.plot(x, y)
plt.title("Line Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


3. Common Plot Types


✅ a) Line Plot


plt.plot(df['year'], df['revenue'])
plt.title('Yearly Revenue')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.grid(True)
plt.show()


✅ b) Bar Chart


categories = ['A', 'B', 'C']
values = [30, 50, 20]
plt.bar(categories, values, color='skyblue')
plt.title('Category Distribution')
plt.show()


✅ c) Histogram


plt.hist(df['age'], bins=10, color='green')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
    

✅ d) Scatter Plot

plt.scatter(df['height'], df['weight'], color='red')
plt.title('Height vs Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()


✅ e) Pie Chart


labels = ['Male', 'Female']
sizes = [60, 40]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Gender Distribution')
plt.show()
    
    


4. Customization Options
Matplotlib allows you to make plots highly customizable:

color, linestyle, linewidth, marker, etc.

display legend with plt.legend()

save as image with plt.savefig("filename.png")


5. Multiple Plots in One Figure

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(df['x'], df['y'])
axs[0, 0].set_title('Line Plot')

axs[0, 1].bar(categories, values)
axs[0, 1].set_title('Bar Chart')

axs[1, 0].hist(df['age'], bins=15)
axs[1, 0].set_title('Histogram')

axs[1, 1].scatter(df['height'], df['weight'])
axs[1, 1].set_title('Scatter Plot')

plt.tight_layout()
plt.show()


#Feature Engineering
Feature Engineering is the step where the performance of the model is improved by generating new information from the data. This step is often called the "secret weapon to modeling success."



🎯 What is Feature Engineering?


-It is a process where:

-New features are created

-Unnecessary features are removed

-Existing features are transformed to make them more useful

-Features are adapted for model training



🔑 Importance of Feature Engineering
✅ Helps to increase model accuracy
✅ Reduces Underfitting / Overfitting
✅ Can reveal hidden patterns in data
✅ Good performance is achieved even with simple models



1️⃣ Feature Creation

Example:

Create a new column by adding two columns
df['total_income'] = df['applicant_income'] + df['coapplicant_income']

New feature from date:
df['application_month'] = pd.to_datetime(df['application_date']).dt.month


2️⃣ Feature Transformation

Log Transformation: To scale large numbers

df['log_income'] = np.log(df['income'] + 1)

Binning: To make continuous values into categories

df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['Teen', 'Adult', 'Middle-age', 'Senior'])

Normalization / Standardization:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['age', 'salary']] = scaler.fit_transform(df[['age', 'salary']])


3️⃣ Encoding Categorical Variables



Label Encoding


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
One-Hot Encoding


df = pd.get_dummies(df, columns=['marital_status'])



4️⃣ Handling Skewness
Most models expect a normal distribution. So highly skewed data is transformed:

from scipy.stats import boxcox
df['transformed'] = boxcox(df['feature'] + 1)[0]



5️⃣ Feature Selection
Not all features are needed. Only important features are kept and the rest are eliminated:

-Correlation Matrix
-Feature Importance (RandomForest/XGBoost)
-Recursive Feature Elimination (RFE)

#coding 
from sklearn.feature_selection import SelectKBest, f_classif
selected = SelectKBest(score_func=f_classif, k=10).fit_transform(X, y)

Introduction to machine learning

A high-level overview of machine learning for people with little or no knowledge of computer science and statistics. You'll learn some essential concepts, explore data, and interactively go through the machine learning lifecycle, using Python to train, save, and use a machine learning model, just like in the real world.



Machine Learning Algorithms Cheat Sheet

1. Linear Regression (Regression)
Use for: Predicting continuous values.


Formula:

y=β0​+β1​x1​+β2​x2​+⋯+βn​xn​
 
Pros: Simple, interpretable.
Cons: Assumes linearity.


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


2. Logistic Regression (Classification)
Use for: Yes/No predictions.


Formula:
 P(y=1)=1/1+e−(β0​+β1​x1​+…)
 
Pros: Works well for binary classification.
Cons: Not great for complex boundaries.

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


3. Decision Tree
Use for: Classification & Regression.
Idea: Split data based on conditions to reduce impurity (Gini, Entropy).


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


4. Random Forest (Ensemble - Bagging)
Use for: Classification & Regression.
Idea: Many decision trees → average/vote results.

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

5. Gradient Boosting / XGBoost (Ensemble - Boosting)
Use for: High accuracy tasks.
Idea: Build trees sequentially to fix previous mistakes.

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

6. K-Nearest Neighbors (KNN)
Use for: Classification & Regression.
Idea: Look at closest K points in feature space.

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)


7. Naive Bayes
Use for: Text classification, spam detection.

Formula:
P(A∣B)=P(B∣A)⋅P(A)​/P(B)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)


8. Support Vector Machine (SVM)
Use for: Classification with margin maximization.

from sklearn.svm import SVC
model = SVC(kernel='rbf')
model.fit(X_train, y_train)


9. K-Means Clustering (Unsupervised)
Use for: Grouping similar data.

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)

10. Principal Component Analysis (PCA)
Use for: Dimensionality reduction.

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)