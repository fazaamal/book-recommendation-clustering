from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
import pandas as pd
import random

# Load the data
df = pd.read_csv('data.csv').dropna(subset=['genre', 'rate', 'rating_count', 'title'])
df['rating_count'] = df['rating_count'].replace(',','', regex=True)

# Function to recommend a book in each genre
def erewew(favorites, genre):
    # Filter the books to only include those in the specified genre
    genre_books = df.copy()
    
    for i in range(10018):
        try:
            if(genre not in genre_books['genre'][i]):
                genre_books = genre_books.drop(i)
        except:
            continue

    ids = genre_books['id'].values.tolist()
    favourites = []
    
    for i in range(3):
        favourites.append(ids[random.randint(0,len(ids)-1)])
    
    print(favourites)
    # Get the ratings of the favorite books
    favorite_ratings = df[df['id'].isin(favorites)][['id', 'rate']]
    
    # Calculate the mean rating of the favorite books
    mean_rating = favorite_ratings['rate'].mean()
    
    
    # Use KMeans to cluster the books based on their ratings and rating counts
    X = genre_books[['rate', 'rating_count']].values
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    
    # Get the cluster labels for each book
    labels = kmeans.labels_
    
    # Create a new column in the dataframe with the cluster labels
    genre_books['cluster'] = labels
    
    # Return the book in the higher-rated cluster that has a rating above the mean rating of the favorite books
    return genre_books[(genre_books['cluster'] == 1) & (genre_books['rate'] > mean_rating)]['id'].values[0]

def recommend_book(favourites):
    
    X = df[['rate', 'rating_count']].values
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    
    labels = kmeans.labels_
    df['cluster'] = labels
    
    print(df)

# Example usage
favorites = [8,9,10]
genre = 'Science Fiction'
recommended_book = recommend_book(favorites)
# print(f'We recommend the book with ID {recommended_book} in the {genre} genre.')
# print(df[df['id'] == 700])