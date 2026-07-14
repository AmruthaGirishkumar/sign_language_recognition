import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

print(" Loading landmark data...")
X = np.load("data/x_data.npy")
y = np.load("data/y_labels.npy")

print(" Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(" Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(" Training Neural Network (MLPClassifier)...")
model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42)
model.fit(X_train, y_train)

print(" Saving model and scaler...")
joblib.dump(model, "model/sign_language_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print(" Training complete.")
print(" Evaluating on test set...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
