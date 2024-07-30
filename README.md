# Loan-Risk-Predicting
Loan risk prediction is a critical process in the financial industry, designed to evaluate the probability that a borrower will default on a loan. This involves a series of steps that leverage data analysis and machine learning to provide accurate risk assessments.
## Table of Contents

1. Introduction 
2. Data Exploration 
3. Feature Engineering 
4. Model Building and Evaluation 
5. Results and Insights 
6. Conclusion

## Technologies Used

• Python 
• Pandas 
• NumPy 
• Scikit-learn 
• Matplotlib 
• Seaborn 

## Data Exploration
The dataset was provided in JSON format and included various financial features. The data exploration process involved understanding the distribution of features, handling missing values, and preparing the data for modeling.

### Data Visualizations
1. **Age Distribution:** The age distribution of clients is visualized to understand the demographic spread.
2. **Income Distribution:** Visualizing the income distribution helps in identifying potential outliers and understanding the overall wealth distribution among clients.
3. **Risk Distribution:** Understanding the distribution of risk flags in the dataset.

### Data Exploration Insights
- **Missing Values:** The dataset contained missing values in both numeric and categorical columns. Missing values were handled by filling numeric columns with their median values and categorical columns with their mode.
- **Outliers:** Identified and handled potential outliers in numeric columns to ensure model robustness.
- Feature Relationships: Explored relationships between features using correlation matrices and scatter plots.

## Feature Engineering
Feature engineering involved creating new features and encoding categorical variables. Significant features such as age, income, and experience were included.

### Steps
1. **Handling Missing Values:** Numeric columns were filled with median values, and categorical columns were filled with mode values.
2. **Categorical Encoding:** Converted categorical variables into dummy variables using one-hot encoding.

## Model Building and Evaluation
A Random Forest Classifier was used to build the prediction model. Hyperparameter tuning was performed using GridSearchCV to optimize the model's performance.

### Hyperparameter Tuning
The parameter grid for GridSearchCV included:
```bash
• n_estimators: [100, 200, 300] 
• max_depth: [None, 10, 20 , 30] 
• min_samples_split: [2, 5, 10] 
• min_samples_leaf: [1, 2, 4]
```
The best parameters found were: 
```bash
• max_depth: None 
• min_samples_leaf: 1 
• min_samples_split: 10 
• n_estimators: 200 
```
### Model Performance 
The model was evaluated on both training and testing data to ensure it generalizes well.
### Training Performance
```bash
• Accuracy: 0.9999603174603174 
• Classification Report: 
            precision    recall  f1-score   support 
 
           0       1.00      1.00      1.00    176857 
           1       1.00      1.00      1.00     24743 
 
    accuracy                           1.00    201600 
   macro avg       1.00      1.00      1.00    201600 
weighted avg       1.00      1.00      1.00    201600 
```
### Testing Performance
```bash
• Testing Accuracy: 0.9014880952380953 
• Testing Classification Report: 
               precision    recall  f1-score   support 
 
           0       0.93      0.96      0.94     44147 
           1       0.64      0.46      0.54      6253 
 
    accuracy                           0.90     50400 
   macro avg       0.78      0.71      0.74     50400 
weighted avg       0.89      0.90      0.89     50400 

• Confusion Matrix: 
[[42538  1609] 
 [ 3356  2897]] 
```
## Results and Insights
### Final Model Performance
``` bash
• Final Testing Accuracy: 0.9045039682539683 
• Final Classification Report: 
 
           precision    recall  f1-score   support 
 
       0       0.92      0.97      0.95     44147 
       1       0.68      0.43      0.53      6253 
 
accuracy                           0.90     50400 
macro avg      0.80      0.70      0.74     50400  
weighted avg   0.89      0.90      0.89     50400 
```
### Main Deciding Factors
The key features influencing the risk prediction include:
 - Employment Status: Stable employment status correlates with 
lower risk.
 - Income: Higher income levels generally indicate lower risk
 - Age Group: Certain age groups may exhibit different risk profiles.

## Conclusion
The Random Forest model effectively predicts loan risk, achieving a testing accuracy of approximately 90%. While the model performs exceptionally well on training data, careful tuning and validation are crucial to prevent overfitting and ensure reliable predictions on new data. The main deciding factors such as credit score and employment status provide valuable insights into the risk assessment process. 
