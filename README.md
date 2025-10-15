# 🧠 Data Science Practice Repository

Welcome to my **Data Science Practice** repository!  
This repo contains my hands-on learning journey, projects, and experiments on **Data Science**, **Machine Learning**, and **Deep Learning** — all developed and tested using **Google Colab** and **VS Code**.

---

## 📘 About This Repository
This repository is a collection of:
- Data preprocessing and feature engineering practices  
- Machine Learning model implementations  
- Deep Learning experiments  
- Visualization and model evaluation codes  
- Deployment examples (Streamlit / Colab)  

---

## 📂 Project Highlights

### 🔹 House Price Prediction 🏠
**Goal:** Predict house prices using the California Housing dataset.  

**Key Steps:**
- Data Cleaning & Feature Engineering  
- Model Training using:
  - Linear Regression  
  - Decision Tree  
  - Random Forest  
  - Gradient Boosting  
  - XGBoost  
- Model Evaluation (MSE & Visualization)
- Feature Importance Plot using Random Forest

📊 **Output Example:**


---

## 🤖 Machine Learning Concepts Covered

- Supervised Learning (Regression & Classification)
- Feature Engineering & Scaling
- Model Evaluation (MSE, RMSE, R²)
- Cross Validation
- Data Visualization using Matplotlib & Seaborn

---

## 🧩 Deep Learning Concepts

- Artificial Neural Networks (ANN)
- Activation Functions
- Optimizers & Loss Functions
- CNN, RNN, LSTM (Coming Soon 🚀)
- TensorFlow / Keras Implementation Examples

---

## 🛠️ Technologies & Tools

| Category | Tools / Libraries |
|-----------|------------------|
| Languages | Python |
| ML/DL | Scikit-Learn, TensorFlow, Keras, XGBoost |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| IDE / Platform | Google Colab, VS Code |
| Deployment | Streamlit |

---

## 📈 Learning Progress

✅ **Completed:**
- Data Science Basics  
- Data Analysis using Pandas & NumPy  
- Feature Engineering  
- Machine Learning Models  
- Power BI for Beginners  
- Introduction to Data Science  

📚 **In Progress:**
- Deep Learning with TensorFlow/Keras  
- Streamlit App Deployment  
- Advanced ML Algorithms  

---

## 📊 Example Visualization

Feature Importance Plot (Random Forest):

```python
def plot_feature_importance(model, features, model_name):
    import numpy as np
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importances ({model_name})')
    plt.bar(range(len(features)), importances[indices], align='center')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


👨‍💻 Author

Md. Ismail Hasan Tanjer
📍 Data Science & Machine Learning Enthusiast
📧 tanjerinfo@gmail.com

🌐 LinkedIn

💻 GitHub
