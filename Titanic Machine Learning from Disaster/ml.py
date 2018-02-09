import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv("data/train.csv", index_col=0)
test_data = pd.read_csv("data/test.csv", index_col=0)
train_data.fillna(-99999, inplace=True)
test_data.fillna(-99999, inplace=True)
feature_cols = ["Sex", "Fare", "Parch", "SibSp"]
X = train_data.loc[:, feature_cols]
y = train_data["Survived"]
test_data = test_data.loc[:, feature_cols]


le = LabelEncoder()
# train_data["Embarked"] = train_data["Embarked"].factorize()[0]
# train_data["Embarked"] = le.fit_transform(train_data["Embarked"])
X["Sex"] = le.fit_transform(X["Sex"])
test_data["Sex"] = le.fit_transform(test_data["Sex"])
# X["Embarked"] = X["Embarked"].factorize()[0]
# test_data["Embarked"] = le.fit_transform(test_data["Embarked"])

logreg = LogisticRegression(n_jobs=-1)
logreg.fit(X, y)
prediction = logreg.predict(test_data)

predicted_data = pd.DataFrame({"PassengerId": test_data.index, "Survived": prediction})
predicted_data.to_csv("data/prediction.csv", index=None)
