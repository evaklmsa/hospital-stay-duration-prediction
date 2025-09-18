# Project Title
Predicting Hospital Admission Duration

## Introduction
This project focuses on developing a machine learning model to predict the duration of hospital stays for patients. By analyzing a comprehensive healthcare dataset, the aim is to identify key factors influencing a patient's length of admission. The resulting predictive model can assist hospital administration in optimizing resource allocation, improving patient care, and enhancing operational efficiency.

## Problem Statement
The primary objective is to create an accurate regression model that can predict a patient's length of stay (in days). The project addresses the challenge of handling a large-scale healthcare dataset with a variety of numerical and categorical features. The key goals are to:
1.  Perform comprehensive exploratory data analysis to understand the distribution of patient demographics, health conditions, and hospital resources.
2.  Preprocess and transform the data, including handling categorical variables and identifying the most relevant features for prediction.
3.  Implement and compare a diverse set of regression algorithms to find the best-performing model.
4.  Validate the model's performance using metrics such as Mean Squared Error (MSE) and R-squared to ensure its reliability for real-world application.

## Dataset Details
The dataset used is a healthcare record containing 500,000 entries with information about various patient and hospital-related attributes.
* **Source**: The data was sourced from a file named "healthcare_data.csv".
* **Data Size**: The dataset contains 500,000 entries and 15 columns.
* **Key Features**: The features include `Available Extra Rooms in Hospital`, `Department`, `Age`, `gender`, `Type of Admission`, `Severity of Illness`, and `Admission_Deposit`.
* **Target Variable**: The target variable is `Stay (in days)`.
* **Data Preparation**: The `patientid` column was dropped as it was an identifier without predictive value. Categorical columns such as `Department`, `gender`, and `Type of Admission` were encoded into a numerical format for model training. The data was split into training and testing sets, and numerical features were scaled to standardize their influence on the models.

## Methods and Algorithms
This project utilizes a variety of machine learning models for regression:
* **Linear Models**:
    * **Linear Regression**: A baseline model to establish foundational performance.
    * **Ridge, Lasso, and ElasticNet**: Regularized linear models to address multicollinearity and prevent overfitting.
* **Tree-Based Models**:
    * **Decision Tree Regressor**: A foundational tree-based model to capture non-linear relationships.
    * **Random Forest Regressor**: An ensemble method that uses multiple decision trees to improve accuracy and reduce variance.
* **Boosting Models**:
    * **AdaBoost Regressor**: An adaptive boosting algorithm to iteratively improve model performance.
    * **Gradient Boosting Regressor**: A powerful boosting algorithm that builds an additive model in a forward stage-wise fashion.
* **Gradient Boosting Library**:
    * **XGBoost Regressor**: A highly optimized and efficient gradient boosting library for improved performance and speed.
* **Model Evaluation**: The models were evaluated using `R-squared` to measure the proportion of variance explained by the model, `Mean Absolute Error` (MAE) to measure average error, and `Mean Squared Error` (MSE) to penalize larger errors.

## Key Results
* **Optimal Model**: The **XGBoost Regressor** and **Gradient Boosting Regressor** models consistently outperformed the other algorithms. They achieved a significantly lower Mean Squared Error (MSE) and higher R-squared score, indicating superior predictive accuracy in determining hospital stay duration.
* **Feature Importance**: The analysis highlighted that the **`Visitors with Patient`** and **`Admission_Deposit`** features were among the strongest predictors of the length of a patient's stay. This suggests that patient support systems and financial factors may be critical indicators for predicting duration.
* **Model Validation**: Cross-validation was performed to ensure the models' robustness and generalizability, confirming that the performance is consistent across different subsets of the data.

## Tech Stack
* **Libraries**:
    * `pandas` and `numpy` for data manipulation and numerical operations.
    * `matplotlib` and `seaborn` for data visualization.
    * `scikit-learn` for data preprocessing, model selection, and performance metrics.
    * `statsmodels` for statistical tests and model diagnostics.
    * `xgboost` for the XGBoost Regressor model.