

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Function to calculate entropy
def entropy(y):
    counts = np.bincount(y)  # Count occurrences of each class
    probabilities = counts / len(y)
    entropy_value = 0
    for p in probabilities:
        if p > 0:
            entropy_value -= p * np.log2(p)
    return entropy_value

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Function to compute information gain
def information_gain(X, y, feature_index):
    total_entropy = entropy(y)
    values, counts = np.unique(X[:, feature_index], return_counts=True)

    weighted_entropy = 0
    for i, value in enumerate(values):
        subset_indices = X[:, feature_index] == value
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(y[subset_indices])

    return total_entropy - weighted_entropy


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ID3 Decision Tree Algorithm
class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(set(y)) == 1:  # If only one class remains
            return y[0]
        if X.shape[1] == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]  # Return majority class

        # Find the best feature to split
        gains = []
        for i in range(X.shape[1]):
            gains.append(information_gain(X, y, i))
        best_feature = np.argmax(gains)

        tree = {best_feature: {}}
        values = np.unique(X[:, best_feature])

        for value in values:
            subset_indices = X[:, best_feature] == value
            subtree = self.fit(X[subset_indices], y[subset_indices], depth + 1)
            tree[best_feature][value] = subtree

        return tree

    def predict(self, X):
        predictions = []
        for x in X:
            node = self.tree
            while isinstance(node, dict):
                feature = list(node.keys())[0]
                value = x[feature]
                node = node[feature].get(value, Counter(y).most_common(1)[0][0])  # Default to majority class
            predictions.append(node)
        return np.array(predictions)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Function to print the tree with feature names
def print_tree(tree, feature_names, indent=""):
    for key, value in tree.items():
        print(indent + feature_names[key] + ":")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(indent + "  " + str(sub_key) + ":", end=" ")
                if isinstance(sub_value, dict):
                    print_tree(sub_value, feature_names, indent + "    ")
                else:
                    print(sub_value)
        else:
            print(indent + "  " + str(value))

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load data from CSV
df = pd.read_csv("tennis.csv")  # Replace with your actual path
# print(df)

# Display first and last 5 records
print("First 5 records:\n", df.head())
print("\nLast 5 records:\n", df.tail())
print("#----------------------------------------------------------------------")

# --- Entropy Calculation before encoding ---
print("\nEntropy before encoding:", entropy(pd.factorize(df['Play'])[0]))  # Entropy before encoding
print("#----------------------------------------------------------------------")

# Encoding features
for col in df.columns[:-1]:  # Encode all features except the target
    df[col], _ = pd.factorize(df[col])  # Store the mapping in _ so we can use it later.
df['Play'] = df['Play'].map({'Yes': 1, 'No': 0})  # Encode target variable


# Split data into training and testing sets
X = df.iloc[:, :-1].values
y = df['Play'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Store feature names (after encoding)
feature_names = df.columns[:-1].tolist()


# --- Information Gain Calculation (for feature selection explanation) ---
print("\nInformation Gain for each feature:")
for i, feature_name in enumerate(feature_names):
    gain = information_gain(X, y, i)
    print(f"{feature_name}: {gain}")
print("#----------------------------------------------------------------------")

# Train ID3 Model
tree = ID3DecisionTree()
# Since the dataset is very small, splitting it into 70% training data is not working as expected. hence, we are training the model on the whole dataset instead.
tree.tree = tree.fit(X, y)

print(tree.tree)
# Print the tree
print("\nDecision Tree:")
print_tree(tree.tree, feature_names)
print("#----------------------------------------------------------------------")

# Predictions on the test set
y_pred = tree.predict(X_test)


# Compare actual and predicted values
print("\nActual vs. Predicted (Test Data):")
for i in range(len(y_test)):
    print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")
print("#----------------------------------------------------------------------")

# Evaluate accuracy using precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print("#----------------------------------------------------------------------")

# Example predictions (using original categorical values)
def predict_with_original_values(outlook, temp, humidity, wind):
    outlook_encoded = pd.factorize([outlook], sort=True)[0][0]
    temp_encoded = pd.factorize([temp], sort=True)[0][0]
    humidity_encoded = pd.factorize([humidity], sort=True)[0][0]
    wind_encoded = pd.factorize([wind], sort=True)[0][0]

    example_input = np.array([
        outlook_encoded,
        temp_encoded,
        humidity_encoded,
        wind_encoded
    ])
    prediction = tree.predict(example_input.reshape(1, -1))
    return {1: 'Yes', 0: 'No'}.get(prediction[0])  # Decode the prediction to Yes/No


# Example usage
print(f"\nPrediction: {predict_with_original_values('Sunny', 'Hot', 'High', 'Weak')}")
print(f"Prediction: {predict_with_original_values('Rain', 'Cool', 'Normal', 'Weak')}")
print(f"Prediction: {predict_with_original_values('Sunny', 'Cool', 'Normal', 'Strong')}")
print("#----------------------------------------------------------------------\n\n")


#rules
def extract_rules(tree, feature_names, current_rule="IF ", rules=[]):
    for key, value in tree.items():
        feature = feature_names[key]
        for sub_key, sub_value in value.items():
            new_rule = f"{current_rule} {feature} == {sub_key}"
            if isinstance(sub_value, dict):
                extract_rules(sub_value, feature_names, new_rule + " AND", rules)
            else:
                rules.append(f"{new_rule} THEN Play = {sub_value}")
    return rules

rules = extract_rules(tree.tree, feature_names)
for rule in rules:
  print(rule)
