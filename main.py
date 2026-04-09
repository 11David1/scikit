import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#load empty model
clf = RandomForestClassifier(random_state=0)

#feature
df = pd.read_csv("creditcard.csv")
X  = df.iloc[:, :-1] # -> alles, bis auf die letze Spalte
#classes for feature
y = df["Class"].values # -> nur die letze Spalte

#data to test
train_X, test_X, train_y, test_y = train_test_split(X,y, random_state=0)

#learn with the 80%
clf.fit(train_X,train_y)

#takes the 20% and classifies them
print(clf.predict(test_X))