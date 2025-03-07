#CORRELATION
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Example DataFrame
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': [1, 3, 5, 7, 9]
})

# Compute correlation matrix
corr = data.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

#HISTOGRAM
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
data = np.random.normal(0, 1, 1000)

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


#PIECHART
import matplotlib.pyplot as plt

# Data
labels = ['Apples', 'Bananas', 'Cherries', 'Dates']
sizes = [15, 30, 45, 10]
colors = ['red', 'yellow', 'pink', 'brown']

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Fruit Distribution')
plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
plt.show()

#BAR
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Example DataFrame
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [10, 20, 15, 25]
})

# Plot bar graph
plt.figure(figsize=(8, 6))
sns.barplot(x='Category', y='Values', data=data, palette='viridis')
plt.title('Bar Graph of Categories')
plt.xlabel('Category')
plt.ylabel('Values')
plt.show()

#LINE
import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Plot line graph
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', color='green', linestyle='-', linewidth=2, markersize=8)
plt.title('Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

#BOXPLOT
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Example DataFrame
data = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Values': [10, 15, 12, 18, 20, 25]
})

# Plot box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='Category', y='Values', data=data, palette='Set2')
plt.title('Box Plot by Category')
plt.xlabel('Category')
plt.ylabel('Values')
plt.show()

#scatter
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
x = np.random.rand(50)
y = np.random.rand(50)

# Plot scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', alpha=0.7, edgecolor='black')
plt.title('Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

#area
import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Plot area plot
plt.figure(figsize=(8, 6))
plt.fill_between(x, y, color='orange', alpha=0.4)
plt.plot(x, y, color='red', marker='o', linestyle='-', linewidth=2, markersize=8)
plt.title('Area Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()


#cleaning
import pandas as pd
import numpy as np

# Sample DataFrame
data = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, np.nan, 8], 'C': [10, 11, 12, 13]}
df = pd.DataFrame(data)

# Replace missing values with the mean of the column
df_filled = df.fillna(df.mean())

print(df_filled)

#DUPLICATES
# Check for duplicates
print(df.duplicated().sum())

# Drop duplicates
df_no_duplicates = df.drop_duplicates()

print(df_no_duplicates)

# Convert a column to a different data type (e.g., to string)
df['A'] = df['A'].astype(str)

print(df.dtypes)

# Rename columns
df_renamed = df.rename(columns={'A': 'Column1', 'B': 'Column2'})

print(df_renamed.columns)

# Group by a column and calculate mean
grouped = df.groupby('A').mean()

print(grouped)

# Handling outliers using Z-score
from scipy import stats

# Calculate Z-scores
z_scores = np.abs(stats.zscore(df))

# Filter out outliers
df_no_outliers = df[(z_scores < 3).all(axis=1)]

print(df_no_outliers)


# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['Category'])

print(df_encoded)

