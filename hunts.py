
import pandas as pd
import numpy as np

# Load Data
df_train = pd.read_csv("/content/Data Mining Lab PS10 dataset2.csv")
df_test = pd.read_csv("/content/Data Mining Lab PS10 dataset1.csv")

df_train['doors'] = df_train['doors'].astype(str)
df_test['doors'] = df_test['doors'].astype(str)

def create_node(attribute=None, value=None, prediction=None, children=None, threshold=None):
    """Create a node dictionary instead of using a class"""
    return {
        'attribute': attribute,
        'value': value,
        'prediction': prediction,
        'children': children if children is not None else {},
        'threshold': threshold  # Store threshold for numerical splits
    }

def is_numerical(series):
    """Check if a Pandas Series is numerical"""
    return pd.api.types.is_numeric_dtype(series)

def build_tree(df, remaining_attributes, parent_value=None):
    """Build the decision tree recursively using sequential Hunt's algorithm"""
    # If all samples belong to one class
    if len(df['acceptance'].unique()) == 1:
        return create_node(value=parent_value, prediction=df['acceptance'].iloc[0])

    # If no attributes left or empty dataset, return majority class
    if not remaining_attributes or df.empty:
        if not df.empty:
            majority_class = df['acceptance'].mode()[0]  # Normal case
        else:
            majority_class = 'unacc'  # Fallback in case of empty dataset
        return create_node(value=parent_value, prediction=majority_class)

    # Take the next attribute in sequence
    current_attribute = remaining_attributes[0]
    node = create_node(attribute=current_attribute, value=parent_value)

    next_attributes = remaining_attributes[1:]

    if is_numerical(df[current_attribute]):
        threshold = df[current_attribute].median()

        left_subset = df[df[current_attribute] <= threshold]
        right_subset = df[df[current_attribute] > threshold]

        node['threshold'] = threshold
        node['children']['<= ' + str(threshold)] = build_tree(left_subset, next_attributes, '<= ' + str(threshold))
        node['children']['> ' + str(threshold)] = build_tree(right_subset, next_attributes, '> ' + str(threshold))

    else:
        for value in df[current_attribute].unique():
            subset = df[df[current_attribute] == value]
            node['children'][value] = build_tree(subset, next_attributes, value)

    return node


def train_tree(df):
    """Train the decision tree automatically from dataset order"""
    # Extract attributes in dataset order (excluding the class label)
    attribute_order = [col for col in df.columns if col != 'acceptance']
    return build_tree(df, attribute_order)

def predict_single(sample, node):
    """Make prediction for a single sample"""
    if node['prediction'] is not None:
        return node['prediction']

    if node['threshold'] is not None:  # Handle numerical attributes
        value = sample[node['attribute']]
        if value <= node['threshold']:
            key = '<= ' + str(node['threshold'])
        else:
            key = '> ' + str(node['threshold'])
    else:
        value = sample[node['attribute']]
        key = value

    if key in node['children']:
        return predict_single(sample, node['children'][key])

    # If unseen value, return majority class from children
    predictions = [child['prediction'] for child in node['children'].values()
                  if child['prediction'] is not None]
    return max(set(predictions), key=predictions.count) if predictions else 'unacc'

def predict(df, tree):
    """Make predictions for multiple samples"""
    return [predict_single(row, tree) for _, row in df.iterrows()]

def print_tree(node, indent=""):
    """Print the decision tree structure"""
    if node['prediction'] is not None:
        print(f"{indent}Prediction: {node['prediction']}")
        return

    print(f"{indent}Split on {node['attribute']}")
    if node['threshold'] is not None:
        print(f"{indent}Threshold: {node['threshold']}")

    for value, child in node['children'].items():
        print(f"{indent}Value = {value}:")
        print_tree(child, indent + "  ")

def evaluate_model(y_true, y_pred):
    """Calculate accuracy of the model"""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)

def train_and_evaluate_sequential(df_train, df_test):
    """Train and evaluate the sequential decision tree"""
    tree = train_tree(df_train)
    y_pred = predict(df_test, tree)
    accuracy = evaluate_model(df_test['acceptance'], y_pred)

    print(f"\nSequential Decision Tree Structure:")
    print_tree(tree)
    print(f"\nModel Accuracy: {accuracy:.3f}")

    return tree, accuracy

print("Training data shape:", df_train.shape)
print("Testing data shape:", df_test.shape)

print("Columns in df_train:", df_train.columns.tolist())
print("Columns in df_test:", df_test.columns.tolist())

print("Column data types in df_train:")
print(df_train.dtypes)

print("\nColumn data types in df_test:")
print(df_test.dtypes)

df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()

print("Available columns in training data:", df_train.columns)
tree, accuracy = train_and_evaluate_sequential(df_train, df_test)
