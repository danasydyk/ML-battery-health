import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("Loading datasets...")
    X_df = pd.read_csv("smartphone_battery_features.csv")
    y_df = pd.read_csv("smartphone_battery_targets.csv")
    
    # Merge datasets on Device_ID
    df = pd.merge(X_df, y_df, on="Device_ID")
    
    # Drop unused columns
    drop_cols = [
        "thermal_stress_index",
        "avg_charging_cycles_per_week",
        "charging_habit_score",
        "Device_ID"
    ]
    df = df.drop(columns=drop_cols)
    
    # Separate features and target
    y_reg = df["current_battery_health_percent"]
    X = df.drop(columns=["current_battery_health_percent", "recommended_action"])
    
    # Perform train-test split (exact replication of the notebook split)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_reg,
        test_size=0.2,
        random_state=42
    )
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include="number").columns
    categorical_cols = X.select_dtypes(exclude="number").columns
    
    print("Numeric columns:", list(numeric_cols))
    print("Categorical columns:", list(categorical_cols))
    
    # Define pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])
    
    # Build complete model pipeline
    reg_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ])
    
    print("Training Gradient Boosting Regressor...")
    reg_model.fit(X_train, y_train)
    
    # Evaluate model
    pred = reg_model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    print("Model Evaluation:")
    print(f"  MAE:  {mae:.5f}")
    print(f"  RMSE: {rmse:.5f}")
    print(f"  R2:   {r2:.5f}")
    
    # Save model pipeline
    model_path = "model.pkl"
    print(f"Saving model pipeline to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(reg_model, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
