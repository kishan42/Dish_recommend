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
MACRO_CALORIES = {"carbs": 4, "protein": 4, "fats": 9}
DAILY_FIBER = {"women": 25, "men": 38}
TEMPERATURE_FACTORS = {
    "Cold (Below 10°C)": 1.2,
    "Moderately Cold (10°C to 18°C)": 1.07,
    "Neutral (18°C to 25°C)": 1.0,
    "Warm (25°C to 30°C)": 1.03,
    "Hot (Above 30°C)": 1.07,
    "Extremely Hot (Above 35°C)": 1.15,
}
time_categories = ["Breakfast", "Lunch", "Snacks", "Dinner"]
ACTIVITY_FACTORS = {
    "Sedentary": 1.2,
    "Lightly active": 1.375,
    "Moderately active": 1.55,
    "Very active": 1.725,
    "Super active": 1.9,
}
EXERCISE_FACTORS = {
    "No exercise": 0,
    "Light exercise": 0.175,
    "Moderate exercise": 0.35,
    "Heavy exercise": 0.525,
    "Very heavy exercise": 0.7,
}
# Map yoga types to exercise factors
YOGA_FACTORS = {
    "None": 0,
    "Gentle Yoga (Hatha, Restorative)": 0.175,
    "Moderate-Intensity Yoga (Power)": 0.35,
    "Intense Yoga (Advanced)": 0.525,
}
# Constants for Macronutrient Ratios
MACRONUTRIENT_RATIOS = {
    "Standard": {
        "Carbs": (0.5, 0.6),      # 50-60%
        "Proteins": (0.1, 0.15),  # 10-15%
        "Fats": (0.2, 0.3),       # 20-30%
    },
    "Diabetic": {
        "Carbs": (0.45, 0.45),    # 45% fixed
        "Proteins": (0.2, 0.2),   # 20% fixed
        "Fats": (0.35, 0.35),     # 35% fixed
    },
}
# Fiber requirements (per meal based on gender)
FIBER_REQUIREMENTS = {
    "Female": (6, 9),  # 6-9 grams per meal
    "Male": (10, 13),  # 10-13 grams per meal
    "Not prefer to say": (6, 13)
}


# Functions
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    elif gender == "Female":
        return (10 * weight) + (6.25 * height) - (5 * age) - 161
    else:
        return ((10 * weight) + (6.25 * height) - (5 * age) + 5) + ((10 * weight) + (6.25 * height) - (5 * age) - 161) / 2

def calculate_tdee(bmr, temp_factor , activity_factor, exercise_factor, goal_factor):
    adjusted_bmr = bmr * temp_factor
    tdee1 = adjusted_bmr * activity_factor
    tdee2 = tdee1 + (exercise_factor * tdee1)
    tdee3 = tdee2 * goal_factor
    return tdee1, tdee2, tdee3

# Function to distribute caloric intake across meals
def calculate_meal_distribution(tdee3):
    # Meal percentage ranges
    meal_percentages = {
        "Breakfast": (0.2, 0.25),  # 20-25%
        "Lunch": (0.3, 0.35),     # 30-35%
        "Snacks": (0.1, 0.15),    # 10-15%
        "Dinner": (0.3, 0.35),    # 30-35%
    }
    
    # Calculate calorie ranges for each meal
    meal_calories = {}
    for meal, (low, high) in meal_percentages.items():
        low_calories = tdee3 * low
        high_calories = tdee3 * high
        meal_calories[meal] = (low_calories, high_calories)
    return meal_calories

# Function to calculate fixed calories based on hunger index
def calculate_fixed_calories(meal_distribution, hunger_level):
    fixed_calories = {}
    for meal, (low, high) in meal_distribution.items():
        if hunger_level == "Low":
            fixed_calories[meal] = low  # Lowest value
        elif hunger_level == "Normal":
            fixed_calories[meal] = (low + high) / 2  # Midpoint value
        elif hunger_level == "High":
            fixed_calories[meal] = high  # Highest value
    return fixed_calories

def calculate_macronutrients(calories, profile_type, gender):
    # Ratios and fiber range for the profile and gender
    ratios = MACRONUTRIENT_RATIOS[profile_type]
    fiber_range = FIBER_REQUIREMENTS[gender]

    # Macronutrient calculations
    carb_kcal = (calories * ratios["Carbs"][0], calories * ratios["Carbs"][1])  # Carbs in kcal
    protein_kcal = (calories * ratios["Proteins"][0], calories * ratios["Proteins"][1])  # Protein in kcal
    fat_kcal = (calories * ratios["Fats"][0], calories * ratios["Fats"][1])  # Fats in kcal

    # Convert kcal to grams
    carbs_grams = [carb_kcal[0] / 4, carb_kcal[1] / 4]  # Carbs in grams
    proteins_grams = [protein_kcal[0] / 4, protein_kcal[1] / 4]  # Proteins in grams
    fats_grams = [fat_kcal[0] / 9, fat_kcal[1] / 9]  # Fats in grams
    fiber_grams = list(fiber_range)  # Convert fiber tuple to list

    # Return formatted dictionary
    return {
        "Carbs (g)": carbs_grams,
        "Proteins (g)": proteins_grams,
        "Fats (g)": fats_grams,
        "Fiber (g)": fiber_grams,
    }
    
def calculate_ranges_for_nutrient(lower_limit, upper_limit, nutrient_type):
    """
    Calculate nutrient-specific ranges for Best, Moderate, and Avoid categories
    based on the provided percentage variations for Carbs, Proteins, and Fats.
    """
    ranges = {}

    # Define variation percentages for each nutrient type
    if nutrient_type == "Carbs":
        best_variation = (-10, 5)
        moderate_lower_variation = (-15, -11)
        moderate_higher_variation = (6, 10)
        avoid_lower_variation = (-16, float("-inf"))
        avoid_higher_variation = (16, float("inf"))
    elif nutrient_type == "Proteins":
        best_variation = (-5, 30)
        moderate_lower_variation = (-10, -6)
        moderate_higher_variation = (31, 45)
        avoid_lower_variation = (-11, float("-inf"))
        avoid_higher_variation = (46, float("inf"))
    elif nutrient_type == "Fats":
        best_variation = (-10, 5)
        moderate_lower_variation = (-15, -11)
        moderate_higher_variation = (6, 10)
        avoid_lower_variation = (-16, float("-inf"))
        avoid_higher_variation = (11, float("inf"))
    else:
        raise ValueError("Invalid nutrient type.")

    # Best range
    ranges["Best"] = (
        lower_limit * (1 + best_variation[0] / 100),
        upper_limit * (1 + best_variation[1] / 100),
    )

    # Moderate ranges
    ranges["M-Lower"] = (
        lower_limit * (1 + moderate_lower_variation[0] / 100),
        lower_limit * (1 + moderate_lower_variation[1] / 100),
    )
    ranges["M-Higher"] = (
        upper_limit * (1 + moderate_higher_variation[0] / 100),
        upper_limit * (1 + moderate_higher_variation[1] / 100),
    )

    # Avoid ranges
    ranges["A-Lower"] = (
        float("-inf"),
        lower_limit * (1 + avoid_lower_variation[0] / 100),
    )
    ranges["A-Higher"] = (
        upper_limit * (1 + avoid_higher_variation[0] / 100),
        float("inf"),
    )

    return ranges

def calculate_all_nutrient_ranges(macronutrients):
    """
    Calculate ranges for all nutrients (Carbs, Proteins, Fats) directly from the macronutrient calculation output.
    """
    nutrient_ranges = {}
    for nutrient, grams_range in macronutrients.items():
        # Skip Fiber as it's not classified (Carbs, Proteins, Fats only)
        if nutrient in ["Carbs (g)", "Proteins (g)", "Fats (g)"]:
            nutrient_type = nutrient.split(" ")[0]  # Extract 'Carbs', 'Proteins', or 'Fats'
            lower_limit, upper_limit = grams_range
            nutrient_ranges[nutrient_type] = calculate_ranges_for_nutrient(lower_limit, upper_limit, nutrient_type)
    return nutrient_ranges

    
def classify_dish_nutrient(dish_value, nutrient_ranges):
    """
    Classify a dish nutrient value into Best, Moderate, or Avoid.
    """
    for category, (low, high) in nutrient_ranges.items():
        if low <= dish_value <= high:
            return category
    return "A-Higher"

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

def classify_dish(dish, user_macronutrient_ranges):
    """
    Classify a dish based on its macronutrient values and the user's ranges.
    """
    results = {}

    for nutrient, ranges in user_macronutrient_ranges.items():
        dish_value = dish["nutrients"].get(nutrient.lower(), 0)
        results[nutrient] = classify_dish_nutrient(dish_value, ranges)

    return {
        "Dish Name": dish["dish_name"],
        "Nutrient Classifications": results,
    }

def recommend_dishes_from_db(collection, user_macronutrient_ranges, selected_time_category):
    """
    Fetch dishes from the database, classify them, and return recommendations.
    """
    # Fetch all dishes
    dishes = fetch_dishes(collection)
    
    print(dishes)
    # Extract relevant data and classify each dish
    recommendations = []
    for dish in dishes:
        processed_dish = extract_relevant_data(dish)
        if not processed_dish["is_verified"]:
            continue  # Skip unverified dishes
        
        # Check if the dish belongs to the selected timing category
        timing_category = dish.get("timing_category", [])
        if selected_time_category not in timing_category:
            continue  # Skip dishes not in the selected category
        
        classified_dish = classify_dish(processed_dish, user_macronutrient_ranges)
        
        # Add relevant fields to the recommendation
        classified_dish.update({
            "Description": dish.get("dish_description", "No description available"),
            "Timing Category": timing_category,
            "Nutrient Info": dish.get("nutrient_info", {}),
            "Ingredient Category": dish.get("ingredient_category", {}),
        })
        
        recommendations.append(classified_dish)

    return recommendations

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
        nutrient_classifications = dish["Nutrient Classifications"]

        # Ensure all required features are available and not "Unknown"
        if "Unknown" in nutrient_classifications.values():
            dish["result"] = "Unknown"
        else:
            # Prepare input features for prediction
            input_features = {
                "Proteins": nutrient_classifications["Proteins"],
                "Carbs": nutrient_classifications["Carbs"],
                "Fats": nutrient_classifications["Fats"]
            }

            # Predict the result
            predicted_result = predict_end_result(input_features)

            # Add the result to the dish
            dish["result"] = predicted_result
        
        updated_dishes.append(dish)
    
    return updated_dishes


# Streamlit Interface
st.title("Recommend Dishes Based on Macronutrient Classifications")

# User Inputs
st.header("User Information")
country_code = st.text_input("Enter your country code (e.g., +91):", value="+91")
gender = st.selectbox("Gender:", ["Male", "Female", "Not prefer to say"])
weight = st.slider("Weight (kg):", min_value=1.0, max_value=300.0, value=70.0, step=0.1)
height = st.slider("Height (cm):", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
dob = st.date_input(
    "Date of Birth:",
    min_value=datetime(1900, 1, 1),  # Minimum selectable year is 1900
    max_value=datetime.now()  # Maximum selectable date is today
)
age = datetime.now().year - dob.year
temperature = st.selectbox("Environment Temperature:", list(TEMPERATURE_FACTORS.keys()))

daily_routine = st.selectbox("Daily Routine:", list(ACTIVITY_FACTORS.keys()))

# Activity Selection
st.header("Physical Activity")
activity_type = st.selectbox("Do you engage in:", ["None", "Exercise", "Yoga"])
selected_activity = None  # Track the specific activity selected

if activity_type == "Exercise":
    selected_activity = st.selectbox("Select your exercise intensity:", list(EXERCISE_FACTORS.keys()))
    activity_factor = EXERCISE_FACTORS[selected_activity]
elif activity_type == "Yoga":
    selected_activity = st.selectbox("Select your yoga intensity:", list(YOGA_FACTORS.keys()))
    activity_factor = YOGA_FACTORS[selected_activity]
else:
    activity_factor = 0


# Goal Consideration Inputs
st.header("Goal Consideration")
goal_subcategory = None  # To store sub-goal details

goal = st.selectbox(
    "Select your goal:",
    ["Healthy Eating", "Muscle Gain", "Fat Loss", "Disease Management (e.g., Diabetes, Cholesterol)"]
)

# Sub-options for Muscle Gain and Fat Loss
if goal == "Muscle Gain":
    goal_subcategory = st.selectbox(
        "Select muscle gain intensity:",
        ["Lean Muscle Gain (Slow and Controlled)", "Moderate Muscle Gain (Balanced Approach)", "Aggressive Muscle Gain (Rapid Bulking)"]
    )
    muscle_gain_factors = {
        "Lean Muscle Gain (Slow and Controlled)": 1.075,
        "Moderate Muscle Gain (Balanced Approach)": 1.175,
        "Aggressive Muscle Gain (Rapid Bulking)": 1.275,
    }
    goal_factor = muscle_gain_factors[goal_subcategory]

elif goal == "Fat Loss":
    goal_subcategory = st.selectbox(
        "Select fat loss intensity:",
        ["Mild Weight Loss (Slow and Sustainable)", "Moderate Weight Loss (Balanced Approach)", "Aggressive Weight Loss (Rapid Results)"]
    )
    fat_loss_factors = {
        "Mild Weight Loss (Slow and Sustainable)": 0.925,
        "Moderate Weight Loss (Balanced Approach)": 0.825,
        "Aggressive Weight Loss (Rapid Results)": 0.725,
    }
    goal_factor = fat_loss_factors[goal_subcategory]
else:
    goal_factor = 1.0

hunger_level = st.selectbox("Hunger Level:", ["Low", "Normal", "High"])

# Calculate BMR
bmr = calculate_bmr(weight, height, age, gender)
st.write(f"BMR: {bmr:.2f} kcal/day")

# Adjusted TDEE
tdee1, tdee2, tdee3 = calculate_tdee(
    bmr,
    TEMPERATURE_FACTORS[temperature],
    ACTIVITY_FACTORS[daily_routine],
    activity_factor,
    goal_factor
)

st.write(f"TDEE1 value : {tdee1:.2f} kcal/day")
st.write(f"TDEE2 value : {tdee2:.2f} kcal/day")
st.write(f"TDEE3 value : {tdee3:.2f} kcal/day")

# Distribute calories for TDEE3
meal_distribution = calculate_meal_distribution(tdee3)

#st.write(f"meal_distribution: {meal_distribution:} kcal/day")

# Get the fixed caloric distribution based on hunger level
fixed_meal_calories = calculate_fixed_calories(meal_distribution, hunger_level)

#st.write(f"fixed_meal_calories: {fixed_meal_calories} kcal/day")


# Display the result for the selected meal category
selected_meal = st.selectbox("Select Time Category:", time_categories)


st.subheader(f"Required Calories Based on {hunger_level} Hunger Level:")
st.write(f"{fixed_meal_calories[selected_meal]:.2f} kcal")

# Streamlit Interface for Macronutrient Ratios
st.header("Macronutrient Ratios for Selected Meal")

# Select user type
profile_type = st.selectbox("Dietary Profile Type:", ["Standard", "Diabetic"])

# Get calories for the selected meal category (from previous step)
selected_meal_calories = fixed_meal_calories[selected_meal]  # Calories for selected meal

# Calculate macronutrients
macronutrients = calculate_macronutrients(selected_meal_calories, profile_type, gender)

# Display results
# User Data JSON
user_data = {
    "Country Code": country_code,
    "Gender": gender,
    "Weight (kg)": weight,
    "Height (cm)": height,
    "Age": age,
    "Temperature": temperature,
    "daily_routine": daily_routine,
    "Activity Type": activity_type,
    "Activity Sub-Category": selected_activity if activity_type != "None" else "None",
    "Goal": goal,
    "Goal Sub-Category": goal_subcategory if goal_subcategory else "None",
    "Hunger Level": hunger_level,
    "Selected Meal": selected_meal,
    "BMR (kcal/day)": bmr,
    "TDEE (kcal/day)": {"Activity Level": tdee1, "Exercise/Yoga Adjusted": tdee2, "Goal Adjusted": tdee3},
    "Hunger Level": hunger_level,
    "Fixed Calories": fixed_meal_calories[selected_meal],
    "profile_type": profile_type,
    "Macronutrient_ratios": macronutrients,
}

# Display JSON
st.header("User Data :")
st.json(user_data)

# Generate Ranges for Each Nutrient
user_macronutrient_ranges = calculate_all_nutrient_ranges(macronutrients)

# Get dish recommendations from the database
recommendations = recommend_dishes_from_db(dish_data, user_macronutrient_ranges,selected_meal)

#print(recommendations)

# Attach predictions to the dishes
updated_dishes = attach_predictions(recommendations)

# Display JSON
st.header("Dish Recommendations :")
st.json(updated_dishes)

# Display Results
print(json.dumps(updated_dishes, indent=4))




