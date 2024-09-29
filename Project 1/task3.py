import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

loan_data = pd.read_csv('loan_data.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(loan_data.head())

# Display the structure of the dataset
print("\nData Structure:")
print(loan_data.info())

# Display summary statistics for numerical features
print("\nSummary Statistics:")
print(loan_data.describe())

# 3. Identify the Target Variable
target_variable = 'Loan_Status'
print(f"\nTarget Variable: {target_variable}")
print("\nUnique values in the target variable:")
print(loan_data[target_variable].unique())

# Display the count of each class in the target variable
print("\nCount of each class in the target variable:")
print(loan_data[target_variable].value_counts())

# 4. Check for Missing Values
missing_values = loan_data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# 5. Visualize Data Distributions
sns.set(style='whitegrid')

# Countplot for loan status
plt.figure(figsize=(8, 5))
sns.countplot(x='Loan_Status', data=loan_data)
plt.title('Loan Status Count')
plt.xlabel('Loan Status (0 = Not Approved, 1 = Approved)')
plt.ylabel('Count')
plt.show()

# Boxplot for numerical features by loan status
numerical_features = loan_data.select_dtypes(include='number').columns.tolist()
plt.figure(figsize=(12, 10))
for i, column in enumerate(numerical_features):
    plt.subplot(3, 2, i + 1)  # Adjust layout as needed
    sns.boxplot(x='Loan_Status', y=column, data=loan_data)
    plt.title(f'Boxplot of {column} by Loan Status')

plt.tight_layout()
plt.show()