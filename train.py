import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib  # We'll use joblib for saving the model directly

print("--- Starting Model Training ---")

# This part remains the same for local tracking
mlflow.set_tracking_uri('file:./mlruns')
with mlflow.start_run() as run:
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)

    # We still log to MLflow for our records
    mlflow.sklearn.log_model(lr, "iris_model")
    print(f"MLflow Run ID for tracking: {run.info.run_id}")

# --- THIS IS THE NEW, IMPORTANT PART ---
# We save the trained model directly to a file named 'model.pkl'
model_filename = 'model.pkl'
joblib.dump(lr, model_filename)

print(f"\nSUCCESS: Model has been saved directly to '{model_filename}'.")
print("This file will be used for deployment.")
print("--- Script Finished ---")


