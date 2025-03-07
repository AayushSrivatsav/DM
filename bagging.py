import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
import seaborn as sns

#INBUILT
# #Step 1: Load the Iris dataset (inbuilt iris dataset)
# df = load_iris()
# # # Step 2: Extract features and target variable
# X, y = df.data, df.target
# print(df)
#--------------------------------------------------------------------

# # Step 1: Load the Iris dataset from the CSV file
df=pd.read_csv("/content/Data Mining Lab PS11 dataset.csv")
# print(df)

# Step 2: Extract features and target variable
X = df.drop(['Id', 'Species'], axis=1)  # Drop the 'Id' and 'Species' columns
y = df['Species']  # Target variable: 'Species'

# Step 3: Split into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Initialize the base classifier (Decision Tree)
base_classifier = DecisionTreeClassifier()

# Step 5: Initialize the BaggingClassifier with 10 estimators
bagging_classifier = BaggingClassifier(
    estimator=base_classifier,  # Decision Tree as base estimator
    n_estimators=10,            # Number of base estimators
    random_state=42             # Random state for reproducibility
)

# Step 6: Train the BaggingClassifier on the training data
bagging_classifier.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

# Step 8: Evaluate the accuracy of the BaggingClassifier, Evaluate using confusion matrix, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
print(f'\nACCURACY: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nCONFUSION MATRIX:")
print(conf_matrix)

# Precision, Recall, and F1-score (for multi-class classification)
precision = precision_score(y_test, y_pred, average='weighted')  # Weighted for multi-class
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# print(f'\nPrecision: {precision:.2f}')
# print(f'\nRecall: {recall:.2f}')
# print(f'\nF1-Score: {f1:.2f}')

print(classification_report(y_test, y_pred))

#Visualizations
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.show()
