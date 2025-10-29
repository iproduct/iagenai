import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Select numeric features
df = pd.read_csv("../data/data_titanic.csv")
numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[numeric_cols]
y = df['Survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_imputed, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_imputed)
score = accuracy_score(y_test, y_pred)

print(score)
