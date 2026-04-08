from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)

# Features: [Hausgröße (m²), Zimmer, Baujahr]

X = [
    [100, 4, 1990], [120, 4, 1995], [140, 5, 2005],
    [160, 6, 2010], [180, 6, 2015], [200, 7, 2020],
    [90, 3, 1980], [110, 4, 1992], [170, 6, 2012],
    [150, 5, 2008]
]

y = [0,0,1,1,2,2,0,0,2,1]

clf.fit(X, y)

X_test = [
    [110, 4, 1990],   # klein, älter -> eher günstig
    [140, 5, 2005],   # durchschnitt -> mittel
    [175, 6, 2015],   # größer, moderner -> mittel/teuer
    [200, 7, 2020],   # groß & neu -> teuer
    [95, 3, 1980]     # klein & alt -> günstig
]

clf.predict(X)  # predict classes of the training data
print(clf.predict(X_test))

print("------------")
print(clf.predict_proba(X_test))

print("------------")
print(clf.feature_importances_) #wie oft & wie stark ein feature zur entscheidung beiträgt