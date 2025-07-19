 🌧️ Rainfall Prediction Using Regression Model

This project predicts rainfall based on historical weather conditions using a **Random Forest Regressor** model. It helps stakeholders like farmers, meteorologists, and disaster management authorities to plan their activities effectively.

Project Overview
**Objective**: To build an accurate and scalable rainfall prediction model using historical weather data.
**Algorithm Used**: Random Forest Regressor (improved over multiple linear regression)
**User Interface**: Simple interactive UI using ipywidgets for live predictions


🔧 Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- ipywidgets
- Jupyter Notebook / Google Colab


 Machine Learning Model

 **Algorithm**: Random Forest Regressor
**Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score
- **Preprocessing**:
  - Scaling with `StandardScaler`
  - One-hot encoding for `Month` and `Year`
  - Pipeline with `ColumnTransformer`


 Features

 Predict rainfall by entering weather parameters like:
  - Temperature
  - Humidity
  - Wind Speed
  - Pressure
  - Cloud Cover
  - Evaporation
   Interactive user input through widgets
  Model evaluation with visualization (scatter plot)


 Sample Inputs

🌡 Temperature (°C): 30.0  
💧 Humidity (%): 80  
🌬 Wind Speed (km/h): 10  
⚖ Pressure (hPa): 1008  
☁ Cloud Cover (%): 60  
💦 Evaporation (mm): 5  
📅 Month: 7  
📆 Year: 2025
