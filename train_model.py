import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

features = np.load("features.npy")
labels = np.load("labels.npy")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Choose classifier
classifier_type = "knn"  # "svm" for SVM
if classifier_type == "knn":
    model = KNeighborsClassifier(n_neighbors=3)
else:
    model = SVC(kernel='linear', probability=True)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

joblib.dump(model, "knn_color_hist_model.pkl")
print("Model saved ✅")
