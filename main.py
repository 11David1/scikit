import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#load empty model
clf = RandomForestClassifier(random_state=0)

#feature
print("load data")
df = pd.read_csv("creditcard.csv")
print("data loaded")
X  = df.iloc[:, :-1] # -> alles, bis auf die letze Spalte
#classes for feature
y = df["Class"].values # -> nur die letze Spalte

#data to test
print("splitting data")
train_X, test_X, train_y, test_y = train_test_split(X,y, random_state=0)

#learn with the 80%
print("model training...")
clf.fit(train_X,train_y)

print("")
print("Done")
print("")

#takes the 20% and classifies them
result = clf.predict(test_X)

c = 0
for i in result:
    if i == 1:
        c += 1
        #print(result[i])
print(c)