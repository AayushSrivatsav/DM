#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:59:09 2025

@author: aayushsrivatsav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Step 1: Load the Iris dataset from the CSV file
data=pd.read_csv("/content/Data Mining Lab PS12 Dataset.csv")
# print(data)

# Step 2: Select relevant columns from the dataset
relevant_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'target']
data = data[relevant_columns]  # Select the columns

# Step 3: Replace '?' with NaN and drop the rows with NaN values
data.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
data.dropna(inplace=True)  # Drop rows with NaN values

# Step 4: Convert 'target' into binary classification (0 = no disease, 1 = disease)
data['target'] = data['target'].map({0: 0, 1: 1})  # Ensure binary labels (if needed)

# Step 5: Split the dataset into features (X) and labels (y)
X = data.drop('target', axis=1)  # Features (drop 'target' column)
y = data['target']  # Labels (target column)

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 6: Initialize the Decision Tree classifier as the base estimator
base_classifier = DecisionTreeClassifier(max_depth=1)  # A shallow tree for weak learner

# Initialize the AdaBoostClassifier with the correct parameter name
adaboost_classifier = AdaBoostClassifier(
    estimator=base_classifier,  # Correct parameter name
    n_estimators=3,  # Number of base classifiers (estimators)
    random_state=42
)

# Train the AdaBoostClassifier on the training data
adaboost_classifier.fit(X_train, y_train)


# Step 7: Print the alpha values of the 3 weak learners
print("Alpha values of the weak learners:")
print(adaboost_classifier.estimator_weights_)


# Step 8: Print the training accuracy of the 3 weak learners
print("\nTraining accuracy of the 3 weak learners:")

for i, estimator in enumerate(adaboost_classifier.estimators_):
    # Get the predictions of each weak learner on the training data
    y_pred_train = estimator.predict(X_train)
    # Calculate accuracy
    accuracy = accuracy_score(y_train, y_pred_train)
    print(f"Weak learner {i+1} - Accuracy: {accuracy:.2f}")


# Step 9: Calculate the final accuracy of the AdaBoost model on test data
y_pred_test = adaboost_classifier.predict(X_test)  # Get predictions on test data
final_accuracy = accuracy_score(y_test, y_pred_test)  # Calculate accuracy

print(f'\nFinal Accuracy of AdaBoost on Test Data: {final_accuracy:.2f}')

print(classification_report(y_test, y_pred_test))