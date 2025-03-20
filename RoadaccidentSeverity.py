import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template

# Load the dataset
df = pd.read_csv("https://github.com/Tanvi-Gajula/Roadaccidentseverity.app/blob/main/RTA%20Dataset.csv")

# convert object type column into datetime datatype column
df['Time'] = pd.to_datetime(df['Time'])

# Extrating 'Hour_of_Day' feature from the Time column
new_df = df.copy()
new_df['Hour_of_Day'] = new_df['Time'].dt.hour
n_df = new_df.drop('Time', axis=1)

# NaN are missing because service info might not be available, we will fill as 'Unknowns'
n_df['Service_year_of_vehicle'] = n_df['Service_year_of_vehicle'].fillna('Unknown')
n_df['Types_of_Junction'] = n_df['Types_of_Junction'].fillna('Unknown')
n_df['Area_accident_occured'] = n_df['Area_accident_occured'].fillna('Unknown')
n_df['Driving_experience'] = n_df['Driving_experience'].fillna('unknown')
n_df['Type_of_vehicle'] = n_df['Type_of_vehicle'].fillna('Other')
n_df['Vehicle_driver_relation'] = n_df['Vehicle_driver_relation'].fillna('Unknown')
n_df['Educational_level'] = n_df['Educational_level'].fillna('Unknown')
n_df['Type_of_collision'] = n_df['Type_of_collision'].fillna('Unknown')

# Define options for dropdown menus
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']
options_types_collision = ['Vehicle with vehicle collision','Collision with roadside objects',
                           'Collision with pedestrians','Rollover','Collision with animals',
                           'Unknown','Collision with roadside-parked vehicles','Fall from vehicles',
                           'Other','With Train']
options_sex = ['Male','Female','Unknown']
options_education_level = ['Junior high school','Elementary school','High school',
                           'Unknown','Above high school','Writing & reading','Illiterate']
options_services_year = ['Unknown','2-5yrs','Above 10yr','5-10yrs','1-2yr','Below 1yr']
options_acc_area = ['Other', 'Office areas', 'Residential areas', 'Church areas',
       'Industrial areas', 'School areas', 'Recreational areas',
       'Outside rural areas', 'Hospital areas', 'Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']

# Initialize Flask app
app = Flask(__name__)

# HTML content for index page
index_html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accident Severity Prediction</title>
</head>
<body>
    <h1>Welcome to the Accident Severity Prediction App 🚧</h1>
    <form action="/predict" method="post">
        <label for="No_vehicles">Number of vehicles involved (1-7): </label>
        <input type="number" name="No_vehicles" id="No_vehicles" min="1" max="7" required><br><br>
        <label for="No_casualties">Number of casualties (1-8): </label>
        <input type="number" name="No_casualties" id="No_casualties" min="1" max="8" required><br><br>
        <label for="Hour">Hour of the day (0-23): </label>
        <input type="number" name="Hour" id="Hour" min="0" max="23" required><br><br>
        <label for="collision">Type of collision: </label>
        <input type="text" name="collision" id="collision" required><br><br>
        <label for="Age_band">Driver age group: </label>
        <input type="text" name="Age_band" id="Age_band" required><br><br>
        <label for="Sex">Sex of the driver: </label>
        <input type="text" name="Sex" id="Sex" required><br><br>
        <label for="Education">Education of driver: </label>
        <input type="text" name="Education" id="Education" required><br><br>
        <label for="service_vehicle">Service year of vehicle: </label>
        <input type="text" name="service_vehicle" id="service_vehicle" required><br><br>
        <label for="Day_week">Day of the week: </label>
        <input type="text" name="Day_week" id="Day_week" required><br><br>
        <label for="Accident_area">Area of accident: </label>
        <input type="text" name="Accident_area" id="Accident_area" required><br><br>
        <button type="submit">Predict</button>
    </form>
</body>
</html>
"""

# HTML content for prediction result
result_html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accident Severity Prediction Result</title>
</head>
<body>
    <h1>Accident Severity Prediction Result</h1>
    <h2>The predicted severity is: {{ severity }}</h2>
</body>
</html>
"""

# Define routes
@app.route('/')
def home():
    return index_html_content

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    No_vehicles = int(request.form['No_vehicles'])
    No_casualties = int(request.form['No_casualties'])
    Hour = int(request.form['Hour'])
    collision = request.form['collision']
    Age_band = request.form['Age_band']
    Sex = request.form['Sex']
    Education = request.form['Education']
    service_vehicle = request.form['service_vehicle']
    Day_week = request.form['Day_week']
    Accident_area = request.form['Accident_area']

    # Random Forest model prediction
    prediction = random_forest_prediction(No_vehicles, No_casualties, Hour, collision, Age_band, Sex, Education, service_vehicle, Day_week, Accident_area)

    # Return prediction result
    return result_html_content.replace('{{ severity }}', prediction)

# Random Forest Prediction Function
def random_forest_prediction(No_vehicles, No_casualties, Hour, collision, Age_band, Sex, Education, service_vehicle, Day_week, Accident_area):
    # Define label encoders for categorical features
    label_encoders = {}
    categorical_columns = ['Day_of_week', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
                           'Service_year_of_vehicle', 'Area_accident_occured', 'Type_of_collision']
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Define features and target variable
    features = ['Number_of_vehicles_involved', 'Number_of_casualties', 'Hour', 'Type_of_collision',
                'Age_band_of_driver', 'Sex_of_driver', 'Educational_level', 'Service_year_of_vehicle',
                'Day_of_week', 'Area_accident_occured']
    target = 'Accident_severity'

    # Train the Random Forest model
    X = df[features]
    y = df[target]
    rf_model = RandomForestClassifier(n_estimators=800, max_depth=20, random_state=42)
    rf_model.fit(X, y)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
