from openvino_kit.sklearn.regression import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# Initialize and train
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
model.evaluate(X_test, y_test)

# Save and reload
model.save_model("rf_model_test.joblib")
model.load_model("rf_model_test.joblib")

# Convert to OpenVINO IR for optimized inference
model.convert_to_ir(X_train, model_name="rf_model")

