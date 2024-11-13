from flask import Flask, request, render_template_string
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['ArogyaDB']
train_collection = db['traindissym']
test_collection = db['testdissym']
precautions_collection = db['Disease_Precaution']
questions_collection = db['Questions_disease']

# Retrieve training and test data from MongoDB collections
train_data = pd.DataFrame(list(train_collection.find()))
test_data = pd.DataFrame(list(test_collection.find()))

# Drop any MongoDB-specific fields
train_data.drop('_id', axis=1, inplace=True)
test_data.drop('_id', axis=1, inplace=True)

# Split the training data into features and target
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

# Split the test data into features and target
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Initialize and train Random Forest model with specified parameters
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=100, min_samples_split=70)
rf_model.fit(X_train, y_train)

# Function to evaluate model and return accuracy
def evaluate_model_accuracy(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

# Function to get disease precautions from MongoDB
def get_precautions(disease):
    precautions_record = precautions_collection.find_one({'Disease': disease})
    if precautions_record:
        precautions = [precautions_record.get(f'Precaution_{i}') for i in range(1, 5) if precautions_record.get(f'Precaution_{i}')]
        return precautions
    return []

# Function to get questions for a disease from MongoDB
def get_questions(disease):
    question_record = questions_collection.find_one({'disease': disease})
    if question_record:
        return question_record.get('questions', [])
    return []

# Function to count co-occurrence of multiple symptoms per disease
def count_cooccurrences(user_symptoms):
    symptom_mask = X_train[[col for col in user_symptoms if col in X_train.columns]].sum(axis=1) == len(user_symptoms)
    disease_count_dict = {disease: 0 for disease in rf_model.classes_}
    for disease in rf_model.classes_:
        count = X_train[symptom_mask & (y_train == disease)].shape[0]
        disease_count_dict[disease] = count
    return disease_count_dict

# Function to predict diseases based on co-occurrence counts
def predict_diseases(user_symptoms):
    disease_counts = count_cooccurrences(user_symptoms)
    filtered_diseases = {disease: count for disease, count in disease_counts.items() if count > 0}
    sorted_diseases = sorted(filtered_diseases.items(), key=lambda item: item[1], reverse=True)
    return sorted_diseases

@app.route('/', methods=['GET', 'POST'])
def interactive_prediction():
    if request.method == 'POST':
        user_symptoms_input = request.form['symptoms']
        user_symptoms_list = [symptom.strip() for symptom in user_symptoms_input.split(',')]
        user_symptoms_list = [symptom for symptom in user_symptoms_list if symptom in X_train.columns]

        possible_diseases = predict_diseases(user_symptoms_list)

        if possible_diseases:
            disease_names = [disease for disease, _ in possible_diseases]
            questions = []
            for disease in disease_names:
                disease_questions = get_questions(disease)
                questions.extend(disease_questions)

            # Render the disease results along with questions
            return render_template_string('''
                <html>
                    <head>
                        <style>
                            body { font-family: Arial, sans-serif; background-color: #f9f9f9; }
                            h1 { color: #333; text-align: center; }
                            form { margin: 20px auto; padding: 20px; border: 1px solid #ccc; background: #fff; max-width: 400px; border-radius: 5px; }
                            label { display: block; margin: 10px 0; }
                            input[type="text"] { width: calc(100% - 20px); padding: 10px; margin: 5px 0 20px; border: 1px solid #ccc; border-radius: 4px; }
                            input[type="submit"] { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
                            input[type="submit"]:hover { background-color: #45a049; }
                            .result-box { background-color: #e7f3fe; border: 1px solid #b3d7ff; border-radius: 5px; padding: 15px; margin-top: 20px; }
                        </style>
                    </head>
                    <body>
                        <h2>Possible Diseases: <strong>{{ ', '.join(disease_names) }}</strong></h2>
                        <h3>Answer the following questions:</h3>
                        <form method="post" action="/result">
                            {% for question in questions %}
                                <label>{{ question }}</label>
                                <input type="text" name="{{ question }}" placeholder="Your answer">
                            {% endfor %}
                            <input type="hidden" name="disease_names" value="{{ ','.join(disease_names) }}">
                            <input type="submit" value="Submit">
                        </form>
                    </body>
                </html>
            ''', questions=questions, disease_names=disease_names)

        return "<h2>No diseases found with the given symptoms.</h2>"

    return '''
        <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; background-color: #f9f9f9; }
                    h1 { color: #333; text-align: center; }
                    form { margin: 20px auto; padding: 20px; border: 1px solid #ccc; background: #fff; max-width: 400px; border-radius: 5px; }
                    label { display: block; margin: 10px 0; }
                    input[type="text"] { width: calc(100% - 20px); padding: 10px; margin: 5px 0 20px; border: 1px solid #ccc; border-radius: 4px; }
                    input[type="submit"] { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
                    input[type="submit"]:hover { background-color: #45a049; }
                </style>
            </head>
            <body>
                <h1> Arogya - The Disease Predictor</h1>
                <form method="post">
                    <label>Enter symptoms separated by commas (e.g., itching, skin_rash):</label>
                    <input type="text" name="symptoms" required>
                    <input type="submit" value="Predict">
                </form>
            </body>
        </html>
    '''

@app.route('/result', methods=['POST'])
def result():
    disease_names = request.form.get('disease_names').split(',')
    responses = {question: request.form[question] for question in request.form if question != 'disease_names'}

    disease_scores = {disease: 0 for disease in disease_names}

    # Count responses for each disease
    for disease in disease_names:
        for question in responses:
            if question in get_questions(disease) and responses[question].strip().lower() in ['yes', 'y', 'yeah', 'yup', 'sure', 'absolutely']:
                disease_scores[disease] += 1  # Increment for each positive response

    # Determine the best disease based on the highest score
    best_disease = max(disease_scores, key=disease_scores.get, default=None)

    # Prepare output for rendering
    output_html = "<div class='result-box'>"
    if best_disease:
        precautions = get_precautions(best_disease)
        output_html += f"<h2>Disease: <strong>{best_disease}</strong></h2>" + \
                       (f"<p>Precautions: {', '.join(precautions) if precautions else 'No precautions found.'}</p>") + \
                       f"<p>Model Accuracy: {evaluate_model_accuracy(rf_model, X_test, y_test):.4f}</p>"
    else:
        output_html += "<h2>No best match found with the given responses.</h2>"
    output_html += "</div>"

    return f'''
        <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; background-color: #f9f9f9; }}
                    h1 {{ color: #333; text-align: center; }}
                    .result-box {{ background-color: #e7f3fe; border: 1px solid #b3d7ff; border-radius: 5px; padding: 15px; margin: 20px auto; max-width: 400px; }}
                    p {{ margin: 5px 0; }}
                </style>
            </head>
            <body>
                <h1>Disease Prediction Result</h1>
                {output_html}
                <a href="/" style="display: block; text-align: center; margin-top: 20px;">Go Back</a>
            </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)