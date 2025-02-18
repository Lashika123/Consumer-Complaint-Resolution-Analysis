# Consumer Complaint Resolution Analysis

## Project Overview
This project focuses on analyzing consumer complaints using machine learning models to predict resolution outcomes. The dataset consists of various customer grievances filed with a company, and the goal is to classify whether a complaint will be resolved efficiently based on different factors such as the nature of the complaint, the company’s response, and customer demographics.

By leveraging data preprocessing, feature engineering, and classification models, this analysis can help businesses improve their complaint resolution strategies and enhance customer satisfaction.

## Features
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling features.
- **Exploratory Data Analysis (EDA)**: Visualizing complaint distribution, response times, and resolution rates.
- **Feature Engineering**: Applying Principal Component Analysis (PCA) and Label Encoding.
- **Model Training & Evaluation**: Comparing multiple classifiers such as Decision Trees, Random Forest, XGBoost, and Logistic Regression.
- **Performance Metrics**: Using accuracy score and confusion matrices to assess model efficiency.

## Installation & Requirements
Ensure you have Python installed along with the required dependencies:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

## Code Explanation
### 1. Import Required Libraries
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
```

### 2. Load and Explore Data
```python
train_data = pd.read_csv("Consumer_Complaints_train.csv")
print(train_data.head())
print(train_data.info())
```
This step loads the dataset and provides an initial overview of the columns, missing values, and data types.

### 3. Data Preprocessing
```python
# Encode categorical features
label_encoder = LabelEncoder()
train_data["Product"] = label_encoder.fit_transform(train_data["Product"])
train_data["Company Response"] = label_encoder.fit_transform(train_data["Company Response"])

# Handle missing values
train_data.fillna(train_data.mean(), inplace=True)
```
Categorical variables such as `Product` and `Company Response` are encoded into numerical values, and missing data is handled appropriately.

### 4. Train-Test Split and Feature Scaling
```python
X = train_data.drop(columns=["Complaint ID", "Resolution"])
y = train_data["Resolution"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
The dataset is split into training and testing sets, and features are standardized for better model performance.

### 5. Train Classification Models
```python
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```
Multiple machine learning models are trained and evaluated based on accuracy scores.

### 6. Model Performance & Insights
- The best-performing model can be selected based on accuracy.
- Confusion matrices can be plotted to analyze misclassification rates.
- Hyperparameter tuning can further optimize model efficiency.

## How to Use
1. Run the notebook in **Jupyter Notebook** or Google Colab.
2. Modify the dataset path if needed.
3. Train different machine learning models and evaluate their performance.
4. Use insights to enhance customer complaint resolution strategies.
