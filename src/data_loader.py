import numpy as np
import pandas as pd


def load_data():
    """Load the 100k ratings data from the files in data/raw"""

    # Load ratings
    ratings = pd.read_csv('../data/raw/u.data', names=['userId', 'itemId', 'rating', 'timestamp'], sep='\t')

    # Load movies data
    movies = pd.read_csv('../data/raw/u.item', names=['movieId', 'title', 'releaseDate', 'videoReleaseDate',
                                                      'imdbUrl', 'unknown', 'action', 'adventure', 'animation',
                                                      'children\'s', 'comedy', 'crime', 'documentary', 'drama', 'fantasy',
                                                      'film-noir','horror', 'musical', 'mystery', 'romance', 'sci-fi',
                                                      'thriller','war', 'western'],
                         sep='|', encoding='latin-1')

    # Load users
    users = pd.read_csv('../data/raw/u.user', names=['userId', 'age', 'gender', 'occupation', 'zipcode'], sep='\t')

    # Load genres
    genres = pd.read_csv('../data/raw/u.genre', names=['genre'], sep='\t')

    # Load occupations
    occupations = pd.read_csv('../data/raw/u.occupation', names=['occupation'], sep='\t')

    return ratings, movies, users, genres, occupations


if __name__ == "__main__":
    # Test the functions
    ratings, movies, users, genres, occupations = load_data()
    print(f"Ratings shape: {ratings.shape}")
    print(f"Movies shape: {movies.shape}")
    print(f"Users shape: {users.shape}")
    print(f"Genres shape: {genres.shape}")
    print(f"Occupations shape: {occupations.shape}")

    print("\nRatings columns:", ratings.columns.tolist())
    print("Movies columns:", movies.columns.tolist())
    print("\nFirst few ratings:")
    print(ratings.head())