import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def hello(event, context):
    
    #data = json.loads(event['body'])
    uuid = event['uuid']
    mealPreference = event['meal_preference']
    gender = event['gender']
    height = int(event['height'])
    weight = int(event['weight'])

    # perform calculations
    bmi = weight / ((height/100) ** 2)

    index = 0
    if bmi <= 17:
        index = 0
    elif bmi > 17 and bmi <= 18.5:
        index = 1
    elif bmi > 18.5 and bmi <= 23:
        index = 2
    elif bmi > 23 and bmi <= 30:
        index = 3
    elif bmi > 30 and bmi <= 35:
        index = 4
    elif bmi > 35:
        index = 5

    maleIndex = 0
    femaleIndex = 0
    if gender == 'Male':
        maleIndex = 1
        femaleIndex = 0
    elif gender == 'Female':
        femaleIndex = 1
        maleIndex = 0

    prediction = trained_model(
        [[height, weight, index, femaleIndex, maleIndex]])
    ingrdient = []
    if (mealPreference == 'vegetarian'):

        if prediction == 0:
            ingrdient = ['Apple', 'Tofu']
        elif prediction == 1:
            ingrdient = ['Brown Rice', 'Oats', 'Milk', 'Sweet Potatoes']
        elif prediction == 2:
            ingrdient = ['Gavar Aur Masoor Ki Dal', 'Jowar Methi Roti']
        elif prediction == 3:
            ingrdient = ['A Vegie Burger With made with a whole grain bun']
        elif prediction == 4:
            ingrdient = ['whole grains', 'legumes',
                         'nuts & seeds', 'avocado', 'olive oil']
        elif prediction == 5:
            ingrdient = ['beans', 'lentils', 'quinoa', 'tofu', 'nuts', 'seeds']

    elif (mealPreference == 'non-vegetarian'):
        if prediction == 0:
            ingrdient = ['Chicken Nuggets', 'Eggs']
        elif prediction == 1:
            ingrdient = ['Eggs', 'Chicken Breast', 'Shrimp']
        elif prediction == 2:
            ingrdient = ['Egg Omelete', 'Chicken Breast',
                         'Shrimp', 'Eggs', 'Turkey Breast']
        elif prediction == 3:
            ingrdient = ['Grilled Chicken Breast',
                         'Eggs Omelete', 'A grilled salmon']
        elif prediction == 4:
            ingrdient = ['poultry', 'fish & lean cuts of red meat']
        elif prediction == 5:
            ingrdient = ['chicken', 'fish', 'turkey', 'lean cuts of beef']
    elif (mealPreference == 'vegan'):
        if prediction == 0:
            ingrdient = ['Nuts','Legumes','Tofu']
        elif prediction == 1:
            ingrdient = ['Brown Rice', 'Whole Wheat Pasta','Oats']
        elif prediction == 2:
            ingrdient = ['Lentils','Chia','Oats']
        elif prediction == 3:
            ingrdient = ['Nut Butter Wrap','Tofu Stir','Avocado']
        elif prediction == 4:
            ingrdient = ['Vegetable Stir','Roasted Sweet Potatoes','Chickpea Salad']
        elif prediction == 5:
            ingrdient = ['beans','lentils','chickpeas','low-sugar protein shake']




    # return results
    

    
    return (
        json.dumps({
            'ingrdient': ingrdient,
        'uuid': uuid,
        'meal_preference': mealPreference}),
        200,
        {'Content-type': "application/json"}
    )
    
def trained_model(fname):
    # Load data from CSV file
    df = pd.read_csv('bmi.csv')

    # Calculate BMI
    # df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

    # Define input and output variables
    X = df[['Gender', 'Height', 'Weight', 'Index']]
    y = df['Nutritant']

    # Convert categorical variables to binary
    X = pd.get_dummies(X, columns=['Gender'])

    print(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # new_data = [[189, 110, 3, 0, 1]]
    y_pred = model.predict(fname)

    print('Predicted nutrition classifications:')
    print(y_pred)

    return y_pred