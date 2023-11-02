# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:50:09 2022

@author: aitza
"""


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Import the dataset from a CSV file
dataset = pd.read_csv(r'D:\Project_103\churn\Churn_Modelling.csv')

# Display basic dataset information
num_rows, num_cols = dataset.shape
print(f'The dataset has {num_rows} rows and {num_cols} columns')
print('==' * 30)
print(' ' * 18, 'Dataset Information')
print('==' * 30)
print(dataset.info())

# Visualize the count of 'Exited' using a bar plot
plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Exited", data=dataset)
for bars in ax.containers:
    ax.bar_label(bars)
plt.savefig('count_of_exited_customers.png')


# Display unique values for each column
print('==' * 30)
print(dataset.nunique())
print('==' * 30)

# Check for duplicate rows
duplicate_rows = dataset.duplicated().sum()
print(f'There are {duplicate_rows} duplicate rows')
print('==' * 30)

# Check for null values in the dataset
null_value_counts = dataset.isnull().sum().rename("Null values")
print(null_value_counts)
print('==' * 30)

# Display unique values and their data types for each column
for col in dataset.columns:
    unique_values = dataset[col].unique()
    data_type = dataset[col].dtype
    print(f"{col}: {unique_values} ({data_type})")

# Drop unnecessary columns
columns_to_drop = ["RowNumber", "CustomerId", "Surname"]
dataset = dataset.drop(columns=columns_to_drop, axis=1)

# Display the shape of the dataset after dropping columns
print(dataset.shape)

# Visualize the correlation matrix heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="cool", vmin=0.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation_matrix_heatmap.png')
plt.show()

# Extract correlations with 'Exited'
correlations_with_exited = correlation_matrix["Exited"]

# Extract positive correlations with 'Exited'
positive_correlations_with_exited = correlations_with_exited[correlations_with_exited > 0].to_frame()

# Plot a histogram for 'Age' with 'Exited' as hue
sns.histplot(data=dataset, x="Age", kde=True, hue="Exited", multiple="dodge")

# Define the bin edges and labels for the age categories
age_bins = [18, 30, 45, 65, 92]
age_labels = ["Young_Adults", "Middle_Aged_Adults", "Old_Adults", "Senior_Citizen"]

# Create the "Age_category" column using pd.cut
dataset["Age_category"] = pd.cut(dataset["Age"], bins=age_bins, labels=age_labels)

# Plot a countplot for 'Age_category' with 'Exited' as hue
ax = sns.countplot(data=dataset, x="Age_category", hue="Exited", palette="Set1")
for bars in ax.containers:
    ax.bar_label(bars)
plt.title("Age category Distribution Vs Exited")

# Plot histograms for 'Balance' and 'EstimatedSalary' with 'Exited' as hue
plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
sns.histplot(data=dataset, x="Balance", kde=True, hue="Exited", multiple="dodge")
plt.subplot(1, 2, 2)
sns.histplot(data=dataset, x="EstimatedSalary", kde=True, hue="Exited", multiple="dodge")
plt.tight_layout()
plt.savefig('histogram_Balance&EstimatedSalary.png')
plt.show()

# Group and count 'EstimatedSalary' by 'Exited'
estimated_salary_counts = dataset.groupby(["Exited"])["EstimatedSalary"].size()

# Display available columns for further analysis
available_columns = dataset.columns

# Define columns to be analyzed as discrete variables
discrete_columns = ["Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"]

# Plot countplots for discrete variables with 'Exited' as hue
plt.figure(figsize=(13, 8))
for i, col in enumerate(discrete_columns):
    plt.subplot(2, 2, i+1)
    ax = sns.countplot(data=dataset, x=col, hue="Exited", palette="Accent")
    for bars in ax.containers:
        ax.bar_label(bars)
    plt.title(f"{col} vs Exited", fontweight="black", size=13, pad=10)
    plt.savefig(f"{col} vs Exited.png")
    plt.tight_layout()

# Extract categorical columns
categorical_columns = dataset.select_dtypes(include="object").columns.tolist()

# Plot countplots for categorical variables with 'Exited' as hue
plt.figure(figsize=(10, 4))
for i, col in enumerate(categorical_columns):
    plt.subplot(1, 2, i+1)
    ax = sns.countplot(data=dataset, x=col, hue="Exited", palette="Dark2")
    for bars in ax.containers:
        ax.bar_label(bars)
    plt.title(f"{col} vs Exited")
    plt.savefig(f"{col} vs Exited.png")
    plt.tight_layout()

# Group and count 'Exited' by 'Gender' and 'Geography'
gender_geography_counts = dataset.groupby(["Gender", "Geography"])["Exited"].value_counts().unstack()

# Group and count 'Exited' by 'Gender' and 'Age_category'
gender_age_category_counts = dataset.groupby(["Gender", "Age_category"])["Exited"].value_counts().unstack()

# Group and count 'HasCrCard' and 'Exited' by 'IsActiveMember'
has_cr_card_and_exited_counts = dataset.groupby("IsActiveMember")[["HasCrCard", "Exited"]].value_counts().unstack()

# Skewness analysis
selected_columns = ["CreditScore", "Age", "EstimatedSalary"]

# Print skewness of selected columns
skewness_values = dataset[selected_columns].skew()
print("Skewness of selected columns:")
print(skewness_values)

# Transform the 'Age' column using the natural logarithm
dataset["Age"] = np.log(dataset["Age"])

# Visualize the distribution of the transformed 'Age' column
sns.histplot(x="Age", kde=True, multiple='dodge', data=dataset)
plt.title("Age Distribution After Transformation")
plt.savefig("Distribution.png")
plt.show()

# Display the first few rows of the dataset
print("First few rows of the dataset after transformation:")
print(dataset.head())

# Perform one-hot encoding for nominal columns
nominal_columns = ["Geography"]
dataset = pd.get_dummies(columns=nominal_columns, data=dataset)

# Display the shape of the dataset after one-hot encoding
print("Shape of the dataset after one-hot encoding:")
print(dataset.shape)

# Label encoding for 'Geography' and 'Gender'
label_encoder = LabelEncoder()
dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])
dataset['Age_category'] = label_encoder.fit_transform(dataset['Age_category'])

# Separating the Dependent & Independent Variables
independent_columns = ["CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
                       "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain", "Age_category"]

dependent_column = "Exited"

# Separate the independent and dependent variables
X = dataset[independent_columns].values
y = dataset[dependent_column].values

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Feature Scaling of Dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instance of 1st Ensemble Model
ensemble_1 = LogisticRegression(random_state=0)

# Instance of 2nd Ensemble Model
ensemble_2 = DecisionTreeClassifier(random_state=0)

# Instance of 3rd Ensemble Model
ensemble_3 = GaussianNB()

# Training the Classifier using Voting_Classifier
classifier = VotingClassifier(
    estimators=[
        ('Logistic_Regression', ensemble_1),
        ('Decision_Tree', ensemble_2),
        ('Gaussian_NB', ensemble_3)
    ],
    voting='soft'
)

classifier.fit(X_train, y_train)

# Prediction on X_Test
predictions_X_test = classifier.predict(X_test)


# Accuracy of Classifier
accuracy = accuracy_score(y_test, predictions_X_test) * 100
print("Accuracy score of Classifier is:", accuracy, "%")

# Classification Report for Voting Classifier
classification_rep = classification_report(y_test, predictions_X_test)
print(f'Classification Report:\n{classification_rep}')

# Training Score Of Classifier
training_score = classifier.score(X_train, y_train)
print("Training Score is:", training_score)

# Testing Score Of Classifier
testing_score = classifier.score(X_test, y_test)
print("Testing Score is:", testing_score)

# Plot the confusion matrix
skplt.metrics.plot_confusion_matrix(y_test, predictions_X_test, normalize=False)

# Display the plot
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()

# Calculate the AUC (Area Under the Curve) for ROC
roc_auc = roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1])
print("AUC (Area Under the Curve) for ROC:", roc_auc)


import joblib

# Save the trained model (VotingClassifier)
model_filename = 'churn_prediction_model.pkl'
joblib.dump(classifier, model_filename)

# Save the StandardScaler
scaler_filename = 'standard_scaler.pkl'
joblib.dump(scaler, scaler_filename)

print(f"Model saved as {model_filename}")
print(f"StandardScaler saved as {scaler_filename}")