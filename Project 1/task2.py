# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_data = sns.load_dataset('iris')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(iris_data.head())

# Display the structure of the dataset
print("\nData Structure:")
print(iris_data.info())

# Display summary statistics for numerical features
print("\nSummary Statistics:")
print(iris_data.describe())

# Check for missing values
missing_values = iris_data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Visualize data distributions
# Countplot for species
plt.figure(figsize=(8, 5))
sns.countplot(x='species', data=iris_data)
plt.title('Count of Each Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()

# Boxplot for numerical features by species
numerical_features = iris_data.select_dtypes(include='number').columns.tolist()
plt.figure(figsize=(12, 10))
for i, column in enumerate(numerical_features):
    plt.subplot(2, 2, i + 1)  # Adjust layout as needed
    sns.boxplot(x='species', y=column, data=iris_data)
    plt.title(f'Boxplot of {column} by Species')

plt.tight_layout()
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(10, 6))

# Select only numeric columns for correlation
numeric_iris_data = iris_data.select_dtypes(include='number')

# Calculate the correlation matrix
correlation = numeric_iris_data.corr()

# Create heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
