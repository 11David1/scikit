from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Daten laden (150 Blumen, 3 Arten)
X, y = load_iris(return_X_y=True)

# 2. Aufteilen: 80% Training, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50
)

# 3. Modell erstellen & trainieren
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 4. Wie gut ist es?
score = model.score(X_test, y_test)
print(f"Genauigkeit: {score:.1%}")  # ~96.7%