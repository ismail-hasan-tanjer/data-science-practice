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

📊 Common Metrics
Task	        Metrics
Classification  -	Accuracy, Precision, Recall, F1-score, AUC
Regression      -	RMSE, MAE, R²

from sklearn.metrics import accuracy_score, mean_squared_error
acc = accuracy_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)



🧠 What is Deep Learning?
🔹 Definition:

Deep Learning (DL) is a subset of Machine Learning (ML) that uses Artificial Neural Networks (ANNs) with many layers (deep networks) to learn patterns from large amounts of data — often without explicit human feature engineering.

➡️ In short:

Machine Learning = Algorithms that learn from data
Deep Learning = Neural networks that automatically learn features from raw data



🧩 How It Works (Conceptually)

A Deep Neural Network (DNN) is inspired by the human brain:
It has neurons (nodes) arranged in layers
Each layer transforms the data and passes it to the next one

example structure: 

Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer


***

Each “neuron” does:

output = activation(weight * input + bias)

Some Code for Practice --

🔹 01_data_analysis.ipynb

import pandas as pd

# Load dataset
df = pd.read_csv('-----------.csv')

# Show first 5 rows
print(df.head())

# Basic info
print(df.info())

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())


🔹 02_data_visualization.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('h--------.csv')

# Pairplot
sns.pairplot(df, hue="species")
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

# Boxplot
sns.boxplot(x='species', y='sepal_length', data=df)
plt.show()



🔹 03_machine_learning.ipynb

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('-----------.csv')

# Split features and target
X = df.drop('species', axis=1)
y = df['species']



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



🔹 04_model_evaluation.ipynb

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

🔹 05_streamlit_deployment.ipynb

# Save as `app.py` if you want to run it with Streamlit
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("🌸 Iris Flower Prediction App")

# User inputs
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.2)

# Load dataset & train model
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
X = df.drop('species', axis=1)
y = df['species']
model = RandomForestClassifier().fit(X, y)

# Prediction
pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
st.write("### 🌼 Predicted Species:", pred[0])



06_data_cleaning.ipynb


import pandas as pd
import numpy as np

# Sample data
data = {
    'Name': ['A', 'B', 'C', np.nan, 'E'],
    'Age': [25, np.nan, 22, 28, 30],
    'Salary': [50000, 54000, np.nan, 62000, 70000]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)


# Fill missing values
df['Name'] = df['Name'].fillna('Unknown')
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())

# Remove duplicates (if any)
df.drop_duplicates(inplace=True)

# Convert data types
df['Age'] = df['Age'].astype(int)

print("\nCleaned Data:\n", df)


🔹 08_linear_regression.ipynb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 6])



# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# Visualization
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.legend()
plt.title("Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


🔹 09_classification.ipynb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# Load data
data = load_breast_cancer()
X, y = data.data, data.target


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))






🔹 10_clustering.ipynb (K-Means)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=200, centers=3, random_state=42)

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_






