import pandas as pd

data = pd.read_csv('Adult Income.csv')
data.isnull().sum(axis=0)
data.dtypes

data_prep = pd.get_dummies(data, drop_first=True)

X = data_prep.iloc[:, :-1]
Y = data_prep.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1234, stratify=Y)

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=1234)
rfc.fit(X_train, y_train)
y_predict = rfc.predict(X_test)

cm2 = confusion_matrix(y_test, y_predict)
score2 = rfc.score(X_test, y_test)
