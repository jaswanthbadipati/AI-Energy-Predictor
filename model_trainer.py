# model_trainer.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# model_trainer.py

def train_and_evaluate(df, target='Global_active_power'):
    features = ['hour', 'day', 'month', 'lag_1']  # Added lag_1
   
    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R² Score:", r2_score(y_test, y_pred))

    # Visualization
    plt.figure(figsize=(12, 4))
    plt.plot(y_test.values[:300], label='Actual')
    plt.plot(y_pred[:300], label='Predicted')
    plt.title("Actual vs Predicted Energy Consumption")
    plt.xlabel("Time Index")
    plt.ylabel("Global Active Power (kilowatts)")
    plt.legend()
    plt.tight_layout()
    plt.show()
