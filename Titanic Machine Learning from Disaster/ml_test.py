import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv("data/train.csv", index_col=0)
train_data.fillna(-99999, inplace=True)
feature_cols = ["Sex", "Pclass", "Fare", "Parch", "SibSp"]
X = train_data.loc[:, feature_cols]
y = train_data["Survived"]

le = LabelEncoder()
# X["Embarked"] = X["Embarked"].factorize()[0]
# X["Embarked"] = le.fit_transform(X["Embarked"])
X["Sex"] = le.fit_transform(X["Sex"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=777)

logreg = LogisticRegression(n_jobs=-1)
logreg.fit(X_train, y_train)
print(logreg.score(X_test, y_test))
