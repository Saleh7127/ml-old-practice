from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from knn_scratch import KNN 

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNN(k=5, distance_metric='manhattan', weighted=True)
knn.fit(X_train_scaled, y_train)

# Predictions and accuracy
predictions = knn.predict(X_test_scaled)
print("Predictions:", predictions)

accuracy = knn.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
