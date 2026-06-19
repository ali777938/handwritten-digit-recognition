import os
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

os.makedirs('models', exist_ok=True)

def train():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', gamma='scale', probability=True, C=5.0))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    joblib.dump(model, 'models/digit_model.pkl')
    print(f'Training complete. Accuracy: {acc:.2f}%')

if __name__ == '__main__':
    train()
