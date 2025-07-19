import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ipywidgets as widgets
from IPython.display import display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load new dataset
file_path = "rainfall_prediction_dataset.csv"  # Replace with your actual file path if needed
df = pd.read_csv(file_path)

# Extract Month and Year from Date
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Define feature columns
categorical_features = ['Month', 'Year']
numeric_features = ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 'Pressure (hPa)',
                    'Cloud Cover (%)', 'Evaporation (mm)']

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Define features and target
X = df[numeric_features + categorical_features]
y = df['Rainfall (mm)']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=1500, max_depth=30, min_samples_split=3,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1))
])
rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)

# Evaluate model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n🌿 Random Forest Model Evaluation:")
print(f"✅ Mean Absolute Error (MAE): {mae_rf:.2f} mm")
print(f"✅ Mean Squared Error (MSE): {mse_rf:.2f}")
print(f"✅ R² Score: {r2_rf:.4f}")

# Scatter plot
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_rf, alpha=0.5, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
plt.xlabel("Actual Rainfall (mm)")
plt.ylabel("Predicted Rainfall (mm)")
plt.title("Actual vs. Predicted Rainfall (Random Forest)")
plt.show()

# User input prediction function
def predict_rainfall(temp, humidity, wind_speed, pressure, cloud_cover, evaporation, month, year):
    input_data = pd.DataFrame([[temp, humidity, wind_speed, pressure, cloud_cover,
                                evaporation, month, year]],
                              columns=['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 'Pressure (hPa)',
                                       'Cloud Cover (%)', 'Evaporation (mm)', 'Month', 'Year'])
    predicted_rain = rf.predict(input_data)[0]
    print(f"\n🌧 Predicted Rainfall: {predicted_rain:.2f} mm")

# Interactive input widgets
temp_input = widgets.FloatText(description="🌡 Temperature (°C):")
humidity_input = widgets.FloatText(description="💧 Humidity (%):")
wind_input = widgets.FloatText(description="🌬 Wind Speed (km/h):")
pressure_input = widgets.FloatText(description="⚖ Pressure (hPa):")
cloud_cover_input = widgets.FloatText(description="☁ Cloud Cover (%):")
evaporation_input = widgets.FloatText(description="💦 Evaporation (mm):")
month_input = widgets.IntSlider(min=1, max=12, description="📅 Month:")
year_input = widgets.IntText(value=2025, description="📆 Year:")
predict_button = widgets.Button(description="Predict Rainfall")

def on_button_click(b):
    predict_rainfall(temp_input.value, humidity_input.value, wind_input.value,
                     pressure_input.value, cloud_cover_input.value, evaporation_input.value,
                     month_input.value, year_input.value)

predict_button.on_click(on_button_click)

# Display UI
display(temp_input, humidity_input, wind_input, pressure_input,
        cloud_cover_input, evaporation_input, month_input, year_input, predict_button)

