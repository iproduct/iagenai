import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Preprocess data
df = pd.read_csv("../data/data_titanic.csv")
X = df.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
y = df['Survived']

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Sex', 'Embarked']),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('passthrough', 'passthrough')
        ]), ['Age', 'Fare', 'SibSp', 'Parch'])
    ])

# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)
