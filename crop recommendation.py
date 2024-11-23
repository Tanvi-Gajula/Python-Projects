import pandas as pd
import numpy as np
import requests

# Load the dataset
file_path = "C:/Users/Tanvi Gajula/Downloads/Crop_recommendation.csv"
data = pd.read_csv(file_path)

# Exclude the last column (crops) from the data
criteria_data = data.iloc[:, :-1]


# Step 1: Get user input for criteria selection
def get_criteria_selection(data):
    available_columns = data.columns
    print(f"Available Criteria: {list(available_columns)}")
    num_criteria = int(input("Enter how many criteria you want to select: "))
    selected_criteria = []
    for i in range(num_criteria):
        criterion = input(f"Select criterion {i + 1}: ")
        if criterion in available_columns:
            selected_criteria.append(criterion)
        else:
            print(f"Invalid criterion. Please select from available criteria: {list(available_columns)}")
    print(f"Selected Criteria: {selected_criteria}")
    return selected_criteria


# Step 2: Get user-provided values for selected criteria
def get_user_values(selected_criteria):
    user_values = {}
    for criterion in selected_criteria:
        value = float(input(f"Enter value for {criterion}: "))
        user_values[criterion] = value
    return user_values


# Step 3: Fuzzify the inputs manually for each criterion
def fuzzify_inputs(selected_criteria, user_values):
    fuzzy_scores = []
    for criterion in selected_criteria:
        criterion_min = data[criterion].min()
        criterion_max = data[criterion].max()
        criterion_avg = (criterion_min + criterion_max) / 2

        # Fuzzification
        user_value = user_values[criterion]
        low_value = max(0, (criterion_avg - user_value) / (criterion_avg - criterion_min))
        medium_value = 1 - abs((user_value - criterion_avg) / (
                    criterion_avg - criterion_min)) if criterion_min < user_value < criterion_max else 0
        high_value = max(0, (user_value - criterion_avg) / (criterion_max - criterion_avg))

        fuzzy_scores.append((low_value, medium_value, high_value))
    return fuzzy_scores


# Step 4: Rank crops based on fuzzy logic and user inputs
def rank_crops_fuzzy(data, selected_criteria, fuzzy_scores):
    data['fuzzy_score'] = 0.0
    for i, criterion in enumerate(selected_criteria):
        criterion_min = data[criterion].min()
        criterion_max = data[criterion].max()
        criterion_avg = (criterion_min + criterion_max) / 2

        low_values = np.maximum(0, (criterion_avg - data[criterion]) / (criterion_avg - criterion_min))
        medium_values = 1 - np.abs((data[criterion] - criterion_avg) / (criterion_avg - criterion_min))
        medium_values[(data[criterion] < criterion_min) | (data[criterion] > criterion_max)] = 0
        high_values = np.maximum(0, (data[criterion] - criterion_avg) / (criterion_max - criterion_avg))

        fuzzy_score = (abs(low_values - fuzzy_scores[i][0]) * 1 +
                       abs(medium_values - fuzzy_scores[i][1]) * 2 +
                       abs(high_values - fuzzy_scores[i][2]) * 3)

        data['fuzzy_score'] += fuzzy_score

    min_fuzzy_score_index = data['fuzzy_score'].idxmin()
    best_crop = data.loc[min_fuzzy_score_index]['label']
    return best_crop


# Function to evaluate accuracy of recommendations against actual crops in the dataset
def evaluate_accuracy(data, selected_criteria):
    correct_recommendations = 0
    total_recommendations = len(data)

    for index in range(total_recommendations):
        row = data.iloc[index]
        user_values = {criterion: row[criterion] for criterion in selected_criteria}
        recommended_crop = rank_crops_fuzzy(data, selected_criteria, fuzzify_inputs(selected_criteria, user_values))

        if recommended_crop == row['label']:
            correct_recommendations += 1

    accuracy = (correct_recommendations / total_recommendations) * 100 if total_recommendations > 0 else 0
    return accuracy


# Function to get air quality data for a specified city
def get_air_quality_data(city):
    url = "https://air-quality-by-api-ninjas.p.rapidapi.com/v1/airquality"
    querystring = {"city": city}
    headers = {
        "x-rapidapi-key": "8bd139777amshc0f0c755c9e9e53p11487fjsn9aa991968e8f",
        "x-rapidapi-host": "air-quality-by-api-ninjas.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        air_quality_data = response.json()
        print(f"\nAir Quality data for {city}:")
        print(
            f"CO Concentration: {air_quality_data['CO']['concentration']} µg/m³, AQI: {air_quality_data['CO']['aqi']}")
        print(
            f"NO2 Concentration: {air_quality_data['NO2']['concentration']} µg/m³, AQI: {air_quality_data['NO2']['aqi']}")
        print(
            f"O3 Concentration: {air_quality_data['O3']['concentration']} µg/m³, AQI: {air_quality_data['O3']['aqi']}")
        print(
            f"SO2 Concentration: {air_quality_data['SO2']['concentration']} µg/m³, AQI: {air_quality_data['SO2']['aqi']}")
        print(
            f"PM2.5 Concentration: {air_quality_data['PM2.5']['concentration']} µg/m³, AQI: {air_quality_data['PM2.5']['aqi']}")
        print(
            f"PM10 Concentration: {air_quality_data['PM10']['concentration']} µg/m³, AQI: {air_quality_data['PM10']['aqi']}")
        print(f"Overall AQI: {air_quality_data['overall_aqi']}")
    else:
        print(f"Error: {response.status_code}, Message: {response.text}")


# Function to get weather data for a specified city
def get_weather_data(city):
    url = "https://openweather43.p.rapidapi.com/weather"
    querystring = {"q": city, "units": "metric"}

    headers = {
        "x-rapidapi-key": "8bd139777amshc0f0c755c9e9e53p11487fjsn9aa991968e8f",
        "x-rapidapi-host": "openweather43.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        weather_data = response.json()
        print(f"\nWeather data for {city}:")
        print(f"Temperature: {weather_data['main']['temp']}°C")
        print(f"Humidity: {weather_data['main']['humidity']}%")
        print(f"Weather: {weather_data['weather'][0]['description'].capitalize()}")
    else:
        print(f"Error: {response.status_code}, Message: {response.text}")


# Main execution flow
if __name__ == "__main__":
    selected_criteria = get_criteria_selection(criteria_data)
    user_values = get_user_values(selected_criteria)

    fuzzy_scores = fuzzify_inputs(selected_criteria, user_values)

    # Rank crops based on criteria and calculated weights for initial recommendation
    recommended_crop = rank_crops_fuzzy(data, selected_criteria, fuzzy_scores)

    # Evaluate recommendations against the entire dataset and get accuracy
    accuracy = evaluate_accuracy(data, selected_criteria)

    # Print recommended crop and accuracy
    print(f"\nRecommended Crop: {recommended_crop}")
    print(f"Accuracy of Recommendations: {accuracy:.2f}%")

    # Ask user for city input where they have land
    city = input(f"Enter the city name where you have land to grow {recommended_crop}: ")

    # Retrieve and display weather and air quality data for the specified city
    get_weather_data(city)
    get_air_quality_data(city)