import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#load empty model
clf = RandomForestClassifier(random_state=0)

#Feature
df = pd.read_csv("creditcard.csv")
X  = df.iloc[:, :-1]

#classes for feature
y = df["Class"].values 

#learn
clf.fit(X,y)

#...jetzt nur noch daten um zu predicten?



