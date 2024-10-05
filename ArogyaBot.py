from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from fuzzywuzzy import process

app = Flask(__name__)

training_dataset = pd.read_csv("C:/Users/Tanvi Gajula/Downloads/ArogyaBot/Training.csv")

X_train = training_dataset.iloc[:, :-1]
y_train = training_dataset.iloc[:, -1]

testing_dataset = pd.read_csv("C:/Users/Tanvi Gajula/Downloads/ArogyaBot/Testing.csv")

X_test = testing_dataset.iloc[:, :-1]
y_test = testing_dataset.iloc[:, -1]

# Assuming X_train.columns are the symptoms
symptom_columns = X_train.columns

# Function to perform fuzzy matching and return the best match and similarity score
def perform_fuzzy_matching(user_symptom, symptom_columns):
    best_match, similarity = process.extractOne(user_symptom, symptom_columns)
    return best_match if similarity >= 80 else None


user_predictions_list = []

models = []
for _ in range(4):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    models.append(model)


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enter Symptoms</title>
    </head>
    <body>
        <h2>Enter Symptoms</h2>
        <form method="post" action="/predict_diseases">
            <input type="text" name="symptoms" placeholder="Enter symptoms, separated by commas"><br><br>
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    '''


@app.route('/predict_diseases', methods=['POST'])
def predict_diseases():
    user_symptoms = request.form['symptoms'].split(',')

    # Use fuzzy matching to find the best-matched symptoms
    matched_symptoms = [perform_fuzzy_matching(symptom, symptom_columns) for symptom in user_symptoms]

    # Remove None values from the matched symptoms
    matched_symptoms = [symptom for symptom in matched_symptoms if symptom is not None]

    symptom_input = [1 if symptom in matched_symptoms else 0 for symptom in symptom_columns]

    user_predictions_list.clear()

    for model in models:
        user_predictions = model.predict([symptom_input])
        user_predictions_list.extend(user_predictions)

    counter = Counter(user_predictions_list)

    y_pred_test = models[0].predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    top_diseases_with_counts = counter.most_common()
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Predicted Diseases with Accuracy</title>
    </head>
    <body>
        <h2>Predicted Diseases: {top_diseases_with_counts}</h2>
        <h3>Accuracy on Testing Set: {accuracy}</h3>
    </body>
    </html>
    '''


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json(silent=True)

    user_input = data['queryResult']['queryText']

    # Split the user input into individual symptoms
    user_symptoms = [symptom.strip() for symptom in user_input.split(',')]

    # Initialize an empty list to store matched symptoms
    matched_symptoms = []

    # Iterate over each symptom provided by the user and find the closest match in the dataset
    for user_symptom in user_symptoms:
        best_match = perform_fuzzy_matching(user_symptom, symptom_columns)
        if best_match:
            matched_symptoms.append(best_match)

    symptom_input = [1 if symptom in matched_symptoms else 0 for symptom in symptom_columns]

    user_predictions_list = []

    for model in models:
        user_predictions = model.predict([symptom_input])
        user_predictions_list.extend(user_predictions)

    counter = Counter(user_predictions_list)

    y_pred_test = models[0].predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    top_diseases_with_counts = counter.most_common()
    response_message = f"The predicted diseases for symptoms {matched_symptoms} are: {top_diseases_with_counts}. Accuracy on Testing Set: {accuracy}"

    return jsonify({"fulfillmentText": response_message})


if __name__ == '__main__':
    app.run(debug=True)