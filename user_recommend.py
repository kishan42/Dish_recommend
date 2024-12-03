import streamlit as st
from datetime import datetime
from math import ceil
import json
import pymongo
import urllib
import pickle
import pandas as pd

username = urllib.parse.quote_plus("Kishan")
password = urllib.parse.quote_plus("KishankFitshield")  # Encode special characters
# MongoDB connection
uri = f"mongodb://{username}:{password}@ec2-13-233-104-209.ap-south-1.compute.amazonaws.com:27017/?authMechanism=SCRAM-SHA-256&authSource=Fitshield"
client = pymongo.MongoClient(uri)
db = client["Fitshield"]  # Database name
#dish_collection = db["Dishdetails"]  # Collection for dish details
dish_data = db["RestroDish"]  # Collection for density values

# Constants
MACRO_CALORIES = {"carbs": 4, "protein": 4, "fats": 9, "fiber": 2 }

# Macronutrient match criteria
macronutrient_criteria = {
    "carbs": {
        "best_match_range": (45, 60),
        "moderate_lower_range": (40, 44),
        "moderate_higher_range": (61, 65),
        "avoid_lower": (None, 39),
        "avoid_higher": (71, None)
    },
    "protein": {
        "best_match_range": (8, 43),
        "moderate_lower_range": (3, 9),
        "moderate_higher_range": (44, 58),
        "avoid_lower": (None, 2),
        "avoid_higher": (59, None)
    },
    "fats": {
        "best_match_range": (15, 30),
        "moderate_lower_range": (10, 14),
        "moderate_higher_range": (31, 35),
        "avoid_lower": (None, 9),
        "avoid_higher": (36, None)
    }
}
    
# Function to calculate nutrient contribution to energy and percentage
def calculate_nutrient_percentage(dish):
    total_energy = next(item['value'] for item in dish['nutrient_info']['macro_nutrients'] if item['name'] == 'energy')
    carbs = next(item['value'] for item in dish['nutrient_info']['macro_nutrients'] if item['name'] == 'carbs')
    protein = next(item['value'] for item in dish['nutrient_info']['macro_nutrients'] if item['name'] == 'proteins')
    fats = next(item['value'] for item in dish['nutrient_info']['macro_nutrients'] if item['name'] == 'fats')
    fiber = next(item['value'] for item in dish['nutrient_info']['macro_nutrients'] if item['name'] == 'fibers')

    # Calculate energy contribution for each nutrient
    energy_from_carbs = carbs * MACRO_CALORIES["carbs"]
    energy_from_protein = protein * MACRO_CALORIES["protein"]
    energy_from_fats = fats * MACRO_CALORIES["fats"]
    energy_from_fiber = fiber * MACRO_CALORIES["fiber"]

    # Calculate percentage contribution to total energy
    percent_carbs = (energy_from_carbs / total_energy) * 100
    percent_protein = (energy_from_protein / total_energy) * 100
    percent_fats = (energy_from_fats / total_energy) * 100
    percent_fiber = (energy_from_fiber / total_energy) * 100

    return {
        "carbs": percent_carbs,
        "protein": percent_protein,
        "fats": percent_fats,
        "fiber": percent_fiber
    }

# Function to classify nutrients based on match criteria
def classify_nutrients(percentages):
    classification = {}

    # Classify Carbs
    if macronutrient_criteria["carbs"]["best_match_range"][0] <= percentages["carbs"] <= macronutrient_criteria["carbs"]["best_match_range"][1]:
        classification["carbs"] = "Best"
    elif macronutrient_criteria["carbs"]["moderate_lower_range"][0] <= percentages["carbs"] <= macronutrient_criteria["carbs"]["moderate_lower_range"][1]:
        classification["carbs"] = "M-Lower"
    elif macronutrient_criteria["carbs"]["moderate_higher_range"][0] <= percentages["carbs"] <= macronutrient_criteria["carbs"]["moderate_higher_range"][1]:
        classification["carbs"] = "M-Higher"
    elif macronutrient_criteria["carbs"]["avoid_lower"][1] > percentages["carbs"]:
        classification["carbs"] = "A-Lower"
    elif percentages["carbs"] > macronutrient_criteria["carbs"]["avoid_higher"][0]:
        classification["carbs"] = "A-Higher"

    # Classify Protein
    if macronutrient_criteria["protein"]["best_match_range"][0] <= percentages["protein"] <= macronutrient_criteria["protein"]["best_match_range"][1]:
        classification["protein"] = "Best"
    elif macronutrient_criteria["protein"]["moderate_lower_range"][0] <= percentages["protein"] <= macronutrient_criteria["protein"]["moderate_lower_range"][1]:
        classification["protein"] = "M-Lower"
    elif macronutrient_criteria["protein"]["moderate_higher_range"][0] <= percentages["protein"] <= macronutrient_criteria["protein"]["moderate_higher_range"][1]:
        classification["protein"] = "M-Higher"
    elif macronutrient_criteria["protein"]["avoid_lower"][1] > percentages["protein"]:
        classification["protein"] = "A-Lower"
    elif percentages["protein"] > macronutrient_criteria["protein"]["avoid_higher"][0]:
        classification["protein"] = "A-Higher"

    # Classify Fats
    if macronutrient_criteria["fats"]["best_match_range"][0] <= percentages["fats"] <= macronutrient_criteria["fats"]["best_match_range"][1]:
        classification["fats"] = "Best"
    elif macronutrient_criteria["fats"]["moderate_lower_range"][0] <= percentages["fats"] <= macronutrient_criteria["fats"]["moderate_lower_range"][1]:
        classification["fats"] = "M-Lower"
    elif macronutrient_criteria["fats"]["moderate_higher_range"][0] <= percentages["fats"] <= macronutrient_criteria["fats"]["moderate_higher_range"][1]:
        classification["fats"] = "M-Higher"
    elif macronutrient_criteria["fats"]["avoid_lower"][1] > percentages["fats"]:
        classification["fats"] = "A-Lower"
    elif percentages["fats"] > macronutrient_criteria["fats"]["avoid_higher"][0]:
        classification["fats"] = "A-Higher"

    return classification

def fetch_dishes(collection):
    """
    Fetch dishes from MongoDB and return only relevant fields.
    """
    # Query MongoDB to fetch relevant dish data
    dishes = collection.find(
        {},  # No filter to fetch all dishes
        {
            "dish_name": 1,
            "dish_description": 1,
            "timing_category": 1,
            "nutrient_info": 1,
            "ingredient_category": 1,
            "is_verified": 1,
            "_id": 0,  # Exclude MongoDB's default ID field
        }
    )
    return list(dishes)

def extract_relevant_data(dish):
    """
    Extract relevant macronutrient values from the dish document.
    """
    macro_nutrients = dish.get("nutrient_info", {}).get("macro_nutrients", [])
    nutrients = {n["name"]: n["value"] for n in macro_nutrients}

    return {
        "dish_name": dish["dish_name"],
        "is_verified": dish.get("is_verified", False),
        "nutrients": {
            "carbs": nutrients.get("carbs", 0),
            "proteins": nutrients.get("proteins", 0),
            "fats": nutrients.get("fats", 0),
        }
    }
    
# Function to fetch dish data from MongoDB and classify
def get_dish_classification():
    """
    Classify a dish based on its macronutrient values and the user's ranges.
    """
    # Fetch all dishes from the collection
    dishes = dish_data.find()

    result_list = []

    for dish in dishes:
        # Calculate nutrient percentages
        nutrient_percentages = calculate_nutrient_percentage(dish)

        # Classify based on percentages
        nutrient_classification = classify_nutrients(nutrient_percentages)

        # Store the final result
        result = {
            "dish_name": dish["dish_name"],
            "micro_nutrients": {
                "carbs": nutrient_percentages["carbs"],
                "protein": nutrient_percentages["protein"],
                "fats": nutrient_percentages["fats"],
                "fiber": nutrient_percentages["fiber"]
            },
            "nutrients_category_classification": nutrient_classification
        }

        result_list.append(result)

    return result_list
    

def predict_end_result(new_features):
    """
    Predicts the End Result for given features.
    
    Parameters:
        new_features (dict): A dictionary with keys 'Proteins', 'Carbs', and 'Fats',
                             and their corresponding categorical values.
                             
    Returns:
        str: The predicted End Result (Best, Moderate, Avoid).
    """
    

    # Load the saved LabelEncoder
    # Load the saved LabelEncoder
    with open("protein_label_encoder.pkl", "rb") as file:
        protein_label_encoder = pickle.load(file)
    # Load the saved LabelEncoder
    with open("carbs_label_encoder.pkl", "rb") as file:
        carbs_label_encoder = pickle.load(file)
        # Load the saved LabelEncoder
    with open("fats_label_encoder.pkl", "rb") as file:
        fats_label_encoder = pickle.load(file)
    # Load the saved LabelEncoder
    with open("end_label_encoder.pkl", "rb") as file:
        end_label_encoder = pickle.load(file)
        
            
    # Load the saved model
    with open("random_forest_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    if new_features['Proteins'] in ['A-Lower', 'A-Higher']:
        new_features['Proteins'] = 'Avoid'

    # Convert new features to a DataFrame
    new_features_df = pd.DataFrame([new_features])
    
    # Encode the new features using the same LabelEncoder as before
    new_features_df['Proteins'] = protein_label_encoder.transform(new_features_df['Proteins'])
    new_features_df['Carbs'] = carbs_label_encoder.transform(new_features_df['Carbs'])
    new_features_df['Fats'] = fats_label_encoder.transform(new_features_df['Fats'])
    # Predict the result
    prediction = model.predict(new_features_df)
    
    # Decode the prediction back to the original label
    prediction_decoded = end_label_encoder.inverse_transform(prediction)
    
    return prediction_decoded[0]

def attach_predictions(dishes):
    """
    Attaches predicted 'result' to each dish based on its macronutrient classifications.
    
    Parameters:
        dishes (list): List of dish dictionaries returned by `recommend_dishes_from_db`.
    
    Returns:
        list: Updated list with 'result' key added to each dish.
    """
    updated_dishes = []

    for dish in dishes:
        nutrient_classifications = dish["nutrients_category_classification"]

        # Ensure all required features are available and not "Unknown"
        if "Unknown" in nutrient_classifications.values():
            dish["result"] = "Unknown"
        else:
            # Prepare input features for prediction
            input_features = {
                "Proteins": nutrient_classifications["protein"],
                "Carbs": nutrient_classifications["carbs"],
                "Fats": nutrient_classifications["fats"]
            }

            # Predict the result
            predicted_result = predict_end_result(input_features)

            # Add the result to the dish
            dish["result"] = predicted_result
        
        updated_dishes.append(dish)
    
    return updated_dishes


# Streamlit Interface
st.title("Recommend Dishes Based on Macronutrient Classifications")

# Fetch dishes data
final_results = get_dish_classification()

# Convert results to DataFrame for easier display
#df = pd.DataFrame(final_results)

#print(recommendations)

# Attach predictions to the dishes
updated_dishes = attach_predictions(final_results)

# Display JSON
st.header("Dish Recommendations :")
st.json(updated_dishes)

# Display Results
print(json.dumps(updated_dishes, indent=4))
