
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('C:/Users/KIIT/Downloads/titanic.csv')

X = dataset.drop('Survived', axis=1)
y = dataset['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

print(f'Training set shape: {train_data.shape}')
print(f'Testing set shape: {test_data.shape}')

train_data.to_csv('C:/Users/KIIT/Downloads/train.csv', index=False)

test_data.to_csv('C:/Users/KIIT/Downloads/test.csv', index=False)

train_df = pd.read_csv('C:/Users/KIIT/Downloads/train.csv')
test_df = pd.read_csv('C:/Users/KIIT/Downloads/test.csv')

print(train_df.head())

print(train_df.isnull().sum())

sns.countplot(x='Survived', data=train_df)
plt.show()

sns.histplot(train_df['Age'].dropna(), kde=True)
plt.show()

sns.countplot(x='Survived', hue='Sex', data=train_df)
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=train_df)
plt.show()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Preprocess the training data
X = train_df.drop(['Survived', 'Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)
y = train_df['Survived']

X['Age'].fillna(X['Age'].median(), inplace=True)
X['Fare'].fillna(X['Fare'].median(), inplace=True)

X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

# Fit scaler and model
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

model = RandomForestClassifier()
model.fit(X_scaled, y)

train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)

train_df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:
{conf_matrix}')
print(f'Classification Report:
{class_report}')

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)
missing_cols = set(X.columns) - set(test_df.columns)

for c in missing_cols:
    test_df[c] = 0
    
test_df = test_df[X.columns]
test_df = scaler.transform(test_df)
test_predictions = model.predict(test_df)

print(test_predictions)
    