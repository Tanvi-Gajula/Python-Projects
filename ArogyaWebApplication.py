import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import networkx as nx

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


# Function to normalize user responses
def normalize_response(response):
    response = response.lower().strip()
    if response in ['yes', 'y', 'yeah', 'yup', 'sure', 'absolutely']:
        return 'yes'
    elif response in ['no', 'n', 'nope', 'nah', 'not really']:
        return 'no'
    return response  # For other responses


# Function to get disease precautions from MongoDB
def get_precautions(disease):
    precautions_record = precautions_collection.find_one({'Disease': disease})
    if precautions_record:
        precautions = [precautions_record.get(f'Precaution_{i}') for i in range(1, 5) if
                       precautions_record.get(f'Precaution_{i}')]
        return precautions
    return []


# Function to get questions for a disease from MongoDB
def get_questions(disease):
    question_record = questions_collection.find_one({'disease': disease})
    if question_record:
        return question_record.get('questions', [])
    return []


# Function to visualize a simple decision tree using Matplotlib and NetworkX
def visualize_tree(symptoms, diseases):
    G = nx.DiGraph()  # Create a directed graph

    # Add nodes for symptoms
    G.add_node("Symptoms")

    for symptom in symptoms:
        G.add_node(symptom)
        G.add_edge("Symptoms", symptom)  # Connect root to symptom

        # Add nodes for diseases associated with each symptom
        for disease in diseases:
            G.add_node(disease)
            G.add_edge(symptom, disease)  # Connect symptom to disease

    # Draw the graph using NetworkX and Matplotlib
    pos = nx.spring_layout(G)  # Positioning of nodes in the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=3000,
            node_color='lightgreen', font_size=10, font_weight='bold',
            edge_color='gray')

    plt.title("Symptom-Disease Decision Tree")
    plt.show()


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


# Function to determine the best disease based on user responses
def get_best_disease(responses, possible_diseases):
    disease_scores = {disease: 0 for disease, _ in possible_diseases}

    for disease in disease_scores.keys():
        positive_responses = responses.get(disease, [])
        disease_scores[disease] = sum(1 for response in positive_responses if response == 'yes')

    # Determine the disease with the highest score
    if disease_scores:
        best_disease = max(disease_scores, key=disease_scores.get)
        return best_disease
    return None


# Main interactive prediction function
def interactive_prediction():
    user_symptoms_input = input("Enter symptoms separated by commas (e.g., itching,skin_rash): ")
    user_symptoms_list = [symptom.strip() for symptom in user_symptoms_input.split(',')]

    # Ensure all user symptoms are present in the dataset columns
    user_symptoms_list = [symptom for symptom in user_symptoms_list if symptom in X_train.columns]

    possible_diseases = predict_diseases(user_symptoms_list)

    if possible_diseases:
        disease_names = [disease for disease, _ in possible_diseases]
        print(f"\nPossible Diseases based on your symptoms: {', '.join(disease_names)}")

        # Visualize the decision tree structure based on symptoms and diseases found.
        visualize_tree(user_symptoms_list, disease_names)

        all_questions = []
        for disease, _ in possible_diseases:
            questions = get_questions(disease)
            if questions:
                all_questions.extend([(disease, question) for question in questions])

        if all_questions:
            print("\nAnswer the following questions (yes/no) for a more accurate diagnosis:")
            responses = {}
            for disease, question in all_questions:
                response = input(f"{question}: ")
                normalized_response = normalize_response(response)
                responses.setdefault(disease, []).append(normalized_response)

            best_disease = get_best_disease(responses, possible_diseases)

            if best_disease:
                precautions = get_precautions(best_disease)
                print(f"\nBest Matching Disease: {best_disease}")
                if precautions:
                    print(f"Precautions: {', '.join(precautions)}")
                else:
                    print("No precautions found for this disease.")
            else:
                print("No best match found with the given symptoms and responses.")

            accuracy = evaluate_model_accuracy(rf_model, X_test, y_test)
            print(f"Model Accuracy: {accuracy:.4f}")
        else:
            print("No questions available for the predicted diseases.")
    else:
        print("No diseases found with the given symptoms.")


# Run interactive prediction function
interactive_prediction()