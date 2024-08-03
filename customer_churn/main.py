import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('data/telco_customer_churn.csv')

# Exploratory Data Analysis (EDA)
print(data.head())
print(data.info())
print(data.describe())

# Visualize Churn
sns.countplot(x='Churn', data=data)
plt.show()

# Data Preprocessing
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data = data.dropna()

# Feature Engineering
data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)
data = pd.get_dummies(data, columns=['InternetService', 'Contract', 'PaymentMethod'])

# Selecting Features and Target
X = data.drop(['Churn', 'customerID'], axis=1)
y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Simple Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
