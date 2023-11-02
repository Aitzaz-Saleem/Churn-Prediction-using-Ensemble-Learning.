# Churn-Prediction-using-Ensemble-Learning.
# Comprehensive Data Analysis and Model Evaluation Report

In this report, we conduct a comprehensive data analysis and model evaluation for a churn prediction problem using a dataset from a CSV file. The main goal is to build a classification model to predict whether customers are likely to churn (Exited = 1) or not (Exited = 0) based on various features.

## Data Analysis

### Dataset Information
- The dataset contains customer information and churn status.
- It consists of 10,000 rows and 12 columns.

### Data Preprocessing
1. **Data Cleaning**:
    - Checked for duplicate rows, and found that there are no duplicate rows.
    - Checked for missing values, and there are no missing values in the dataset.

2. **Feature Selection**:
    - Dropped unnecessary columns: 'RowNumber', 'CustomerId', and 'Surname'.

3. **Exploratory Data Analysis (EDA)**:
    - Visualized the distribution of the 'Exited' target variable using a bar plot.
    - Explored the unique values and data types of each column.
    - Visualized the correlation matrix heatmap.
    - Analyzed correlations between features and the 'Exited' target variable.
    - Created age categories and visualized the distribution of 'Exited' within these categories.
    - Visualized histograms for 'Balance' and 'EstimatedSalary' with 'Exited' as hue.
    - Analyzed discrete and categorical variables with 'Exited' as hue.
    - Transformed the 'Age' column using natural logarithm.
    - Performed one-hot encoding for the 'Geography' column.
    - Label encoded 'Gender' and 'Age_category' columns.

### Data Transformation
- Applied feature scaling to the dataset using StandardScaler.

## Model Building and Evaluation

### Ensemble Model
- We built an ensemble model using the VotingClassifier, combining three different base models:
    - Logistic Regression
    - Decision Tree Classifier
    - Gaussian Naive Bayes

### Model Evaluation
- The model was evaluated using the following metrics:
    - Confusion Matrix
        - Non-normalized and normalized confusion matrices were plotted.
    - Accuracy Score: 83.65%
    - Classification Report
        - Detailed classification report showing precision, recall, and F1-score.
    - Training Score: 93.3%
    - Testing Score: 83.65%

### Receiver Operating Characteristic (ROC) Analysis
- We analyzed the model's ROC curve:
    - AUC (Area Under the Curve) for ROC: 0.82
    - The ROC curve visually demonstrates the model's performance in terms of true positive rate and false positive rate.

## Conclusion

In this comprehensive data analysis and model evaluation, we successfully built an ensemble model for churn prediction. The model achieved an AUC of 0.82, indicating good predictive power. The training score was 93.3%, suggesting that the model performed well on the training data. The testing score of 83.65% shows that the model generalizes effectively to unseen data.

This model can be utilized by businesses to identify customers who are at risk of churning and take proactive measures to retain them. Further model optimization and fine-tuning may improve its performance, but the current model provides a good starting point for churn prediction.
