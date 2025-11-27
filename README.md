# Loan-Status-Prediction-using-Machine-Learning-SVM-in-Python
## 📌 Project Overview
##### This project focuses on building a machine learning model that predicts whether a loan application will be approved or rejected based on applicant and financial information. Using the Loan Prediction Dataset, this project includes data preprocessing, exploratory data analysis, label encoding of categorical variables, and training a Support Vector Machine (SVM) classifier with a linear kernel. The workflow includes splitting the dataset into training and testing sets, converting categorical features into numerical values, evaluating accuracy, visualizing results using a confusion matrix, and creating a predictive system. This project demonstrates a complete end-to-end pipeline for solving a real-world classification problem.

## 🎯 Objectives
##### Loan Approval Classification – Build an SVM model to predict whether a loan application is approved.
##### Data Exploration & Preprocessing – Analyze the dataset, handle missing values, and encode categorical variables.
##### Model Development – Train an SVM classifier (kernel='linear') using applicant information.
##### Performance Evaluation – Evaluate performance using accuracy, confusion matrix, and classification report.
##### Predictive System Creation – Develop a system that inputs customer/application features and outputs whether the loan is Granted (1) or Not Granted (0).

## 📁 Dataset
##### Name: Loan Prediction Dataset
##### Source: https://www.kaggle.com/datasets/ninzaami/loan-predication
##### Shape: Rows: 614 samples, Columns: 13 features
#### Target Label:
##### 1 → Loan Approved
##### 0 → Loan Not Approved

## ⚙️ Methodology
#### 1. Data Preprocessing
#### Loaded dataset and checked rows, columns, and missing values.

##### Performed label encoding on categorical features

#### Gender
##### Married
##### Education
##### Replaced target variable Loan_Status:
##### 'Y' → 1
##### 'N' → 0

#### 2. Exploratory Data Analysis (EDA)
##### Visualized relationships using countplot for:
##### Gender vs Loan Approval
##### Education vs Loan Approval
##### Marital Status vs Loan Approval
##### Inspected distributions and approval patterns across categories.

#### 3. Train-Test Split
##### Ensures balanced distribution of loan approvals in both sets.

#### 4. Model Development (SVM)
##### Trained a Support Vector Machine classifier with a linear kernel to classify whether an applicant's loan should be approved or not.
##### Fitted the model using X_train and Y_train.

#### 6. Model Evaluation
##### Calculated model performance metrics, including Training Accuracy and Testing Accuracy. Evaluated prediction behavior with a Confusion Matrix and Classification Report.
##### Visualized the confusion matrix using a heatmap for clear interpretation of true vs. false predictions.

#### 8. Predictive System Creation
##### Developed a simple prediction system where new applicant information (such as income, credit history, loan amount, dependents, and property area) can be input to predict the loan approval status (1 → Loan Granted, 0 → Loan Not Granted).

## 🚀 Results & Insights
#### Model Accuracy
##### Training Accuracy: 79.86%
##### Testing Accuracy: 83.33%
#### Prediction System
##### Successfully predicts loan status for new applicants using only their profile & financial indicators.
#### Model Reliability
##### Small difference between training and testing accuracy → indicates good generalization.
##### The linear SVM performed well because the dataset was relatively well-separable after encoding.

## 🛠 Tech Stack
##### Python – Core programming language
##### NumPy – Numerical computations
##### Pandas – Data loading and preprocessing
##### Scikit-learn – SVM model, train-test split, evaluation metrics
##### Seaborn & Matplotlib – Visualizations (confusion matrix, count plots)
