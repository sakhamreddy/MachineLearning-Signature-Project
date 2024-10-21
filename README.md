SEER Breast Cancer Prediction Project
This repository contains a project focused on developing machine learning models to predict survival outcomes for breast cancer patients using the SEER Breast Cancer dataset. The goal is to use various classification algorithms to accurately predict whether a patient is likely to survive based on their medical history and cancer-related data.

Project Overview
Objective:
The primary objective of this project is to explore the SEER Breast Cancer dataset and develop a predictive model that can classify whether a patient is "Alive" or "Dead" based on their clinical features. This is a binary classification problem.

Dataset:
The dataset used in this project is sourced from the SEER Breast Cancer dataset, containing information about over 4,000 breast cancer patients.
It includes features such as Age, Marital Status, Tumor Size, Survival Months, and other clinical factors.
Techniques Used:
Data Preprocessing: Cleaning, handling missing values, and transforming the data (e.g., normalization, outlier removal).
Exploratory Data Analysis (EDA): Visualizations such as histograms and bar plots to understand feature distributions and relationships with the target variable.
Feature Engineering: Created new features and transformed the data to improve model performance.
Machine Learning Models: Implemented various models including:
Support Vector Machines (SVM)
Decision Trees
Logistic Regression
Random Forest
Bagging and Ensemble techniques
Model Evaluation: Performed evaluation using holdout method and k-fold cross-validation to assess the performance of each model in terms of accuracy, precision, recall, and F-score.
Results:
The Random Forest and Decision Tree models performed best in predicting patient outcomes with balanced data.
We explored Bagging and Ensemble Learning techniques to improve overall model accuracy.
The final decision tree model with balanced data showed a kappa value of 0.8, making it the optimal choice for this classification task.
Business Use Case:
This project demonstrates how machine learning can be applied to breast cancer data to assist in predicting survival outcomes. Healthcare providers can leverage these models to identify high-risk patients and develop personalized treatment plans, contributing to early diagnosis and better decision-making in patient care.

Tools and Libraries:
Programming Languages: R
Libraries: dplyr, ggplot2, corrplot, caret, randomForest, C50, e1071, and others.
Algorithms: Random Forest, SVM, Decision Tree, Logistic Regression, Bagging, Ensemble Models

References:
Rabiei, R., Ayyoubzadeh, S. M., Sohrabei, S., Esmaeili, M., & Atashi, A. (2022). Prediction of breast cancer using machine learning approaches. Journal of Biomedical Physics & Engineering. Link to Article
Nasser, M., & Yusof, U. K. (2023). Deep learning based methods for breast cancer diagnosis: A systematic review and future direction. Diagnostics (Basel). Link to Article
