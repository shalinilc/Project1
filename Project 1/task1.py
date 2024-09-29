# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Titanic Dataset
titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(titanic_url)

# 2. Explore the Dataset
print("Data Structure:")
print(titanic_data.info())
print("\nFirst 5 rows of the dataset:")
print(titanic_data.head())

# 3. Clean the Data
titanic_data = titanic_data.drop(columns=['Ticket', 'Cabin', 'Name'])

# Convert 'Sex' and 'Embarked' to categorical variables
titanic_data['Sex'] = titanic_data['Sex'].astype('category')
titanic_data['Embarked'] = titanic_data['Embarked'].astype('category')

# 4. Handle Missing Values
missing_values = titanic_data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Fill missing Age with median and Embarked with mode
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# Verify if there are any missing values left
missing_values_after = titanic_data.isnull().sum()
print("\nMissing values after handling:")
print(missing_values_after)

# 5. Basic Statistical Analysis
print("\nSummary Statistics:")
print(titanic_data.describe(include='all'))

# 6. Visualize Data Distributions and Correlations
sns.set(style='whitegrid')

# Countplot for survival
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=titanic_data)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Boxplot for Age by Survival
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Age', data=titanic_data)
plt.title('Age Distribution by Survival')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')
plt.show()

# Countplot for survival by gender
plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))

# Select only numeric columns for correlation
numeric_titanic_data = titanic_data.select_dtypes(include='number')

# Calculate correlation matrix
correlation = numeric_titanic_data.corr()

# Create heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(titanic_data, hue='Survived')
plt.title('Pairplot of Titanic Dataset')
plt.show()