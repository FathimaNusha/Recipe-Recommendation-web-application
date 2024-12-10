from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)

# Load the recipe dataset
recipe_df = pd.read_csv(r'recipe_dataset.csv')

# Preprocess ingredients with TF-IDF
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(recipe_df['Ingredients'])

# Train the NearestNeighbors model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_ingredients)

# Function to recommend recipes based on ingredients
def recommend_recipes(input_ingredients):
    if not input_ingredients:
        return []  # Return empty if input is empty 
    
    # Create a regular expression pattern for incredient matching
    pattern = rf'{re.escape(input_ingredients)}'

    # Preprocess input to check if it exists in any recipe's ingredients
    matching_recipes = recipe_df[
        recipe_df['Ingredients'].str.contains(pattern, case=False, na=False)
    ]

    if matching_recipes.empty:
        return []  # Return empty if no match for input ingredients in the dataset

    #else
    # Transform input ingredients using vectorizer
    input_transformed = vectorizer.transform([input_ingredients])
    
    # Get nearest neighbors
    distances, indices = knn.kneighbors(input_transformed)
    
    # Check if the closest recipe is too far (no meaningful match)
    if distances[0][0] > 1.2:  
        return []  # No meaningful matches

    # Retrieve recommended recipes
    recommendations = recipe_df.iloc[indices[0]].copy()
    return recommendations[['Recipe Name', 'Ingredients', 'Cooking Time (minutes)', 'Steps','image-url']].to_dict(orient='records')


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []  # Default to no recommendations
    if request.method == 'POST':
        ingredients = request.form.get('ingredients').strip()  # Get input ingredients
        recommendations = recommend_recipes(ingredients)  # Fetch recommendations

    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
