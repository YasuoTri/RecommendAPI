import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from surprise import SVD, Dataset, Reader
import pickle
import os
import logging
from datetime import datetime
import networkx as nx
from collections import Counter
from surprise.model_selection import GridSearchCV
import random
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_difficulty(difficulty):
    """Normalize course difficulty levels to a score."""
    mapping = {'Beginner': 0.3, 'Intermediate': 0.6, 'Advanced': 0.9}
    return mapping.get(difficulty, 0.3)

# Define level hierarchy for comparison
LEVEL_HIERARCHY = {
    'beginner level': 2,
    'intermediate level': 3,
    'expert level': 4,
    'all levels': 1,
    'unknown': 0
}

def get_level_value(level):
    """Convert course level to numerical value based on hierarchy."""
    return LEVEL_HIERARCHY.get(level.lower().strip(), 0)

# def recommend_similar_courses(course_title, data_file='Data/udemy_courses.csv', num_recommendations=20):
#     """
#     Recommend courses similar to the input course based on course_title, level, and subject.
    
#     Parameters:
#     - course_title (str): Title of the input course.
#     - data_file (str): Path to the CSV file containing course data.
#     - num_recommendations (int): Number of courses to recommend.
    
#     Returns:
#     - list: List of dictionaries containing recommended course details.
#     """
#     try:
#         # Load precomputed models
#         with open('models/courses.pkl', 'rb') as f:
#             df = pickle.load(f)
#         with open('models/tfidf_vectorizer.pkl', 'rb') as f:
#             vectorizer = pickle.load(f)
#         with open('models/tfidf_matrix.pkl', 'rb') as f:
#             tfidf_matrix = pickle.load(f)
#         logger.info(f"Loaded {len(df)} courses and precomputed TF-IDF matrix")
        
#         # Clean data
#         df['course_title'] = df['course_title'].fillna('')
#         df['level'] = df['level'].str.lower().fillna('unknown')
#         df['subject'] = df['subject'].str.lower().fillna('unknown')
        
#         # Get input course details
#         input_course = df[df['course_title'] == course_title].iloc[0]
#         input_level_value = get_level_value(input_course['level'])
#         input_text = ' '.join([
#             input_course['course_title'].lower(),
#             input_course['level'].lower(),
#             input_course['subject'].lower()
#         ])
        
#         # Compute input vector and similarity
#         input_vector = vectorizer.transform([input_text])
#         similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
#         logger.info(f"Similarity scores range: {similarity_scores.min()} to {similarity_scores.max()}")
        
#         # Create similarity DataFrame
#         similarity_df = pd.DataFrame({
#             'course_id': df['course_id'],
#             'course_title': df['course_title'],
#             'similarity': similarity_scores,
#             'level': df['level'],
#             'num_reviews': df['num_reviews'],
#             'num_subscribers': df['num_subscribers']
#         })
        
#         # Filter candidates
#         candidates = similarity_df[
#             (similarity_df['course_title'] != course_title) &
#             (similarity_df['similarity'] > 0)
#         ]
#         logger.info(f"Number of candidates after filtering: {len(candidates)}")
        
#         if candidates.empty:
#             logger.warning("No candidates found after filtering.")
#             return [{"warning": "No similar courses found due to filtering."}]
        
#         # Apply level filter
#         candidates = candidates.copy()
#         candidates['level_value'] = candidates['level'].apply(get_level_value)
#         candidates = candidates[candidates['level_value'] >= input_level_value]
#         logger.info(f"Number of candidates after level filtering: {len(candidates)}")
        
#         if candidates.empty:
#             logger.warning("No candidates found after level filtering.")
#             return [{"warning": "No courses found with level equal to or higher than the input course."}]
        
#         # Sort by similarity and select top candidates
#         top_candidates = candidates.sort_values(by='similarity', ascending=False).head(num_recommendations)
#         logger.info(f"Top candidates: {top_candidates[['course_title', 'similarity', 'level']].to_dict('records')}")
        
#         # Format output
#         recommendations = []
#         for _, row in top_candidates.iterrows():
#             course_row = df[df['course_title'] == row['course_title']].iloc[0]
#             recommendations.append({
#                 'course_id': str(course_row['course_id']),
#                 'course_title': str(course_row['course_title']),
#                 'url': str(course_row['url']),
#                 'is_paid': bool(course_row['is_paid']),
#                 'price': str(course_row['price']),
#                 'num_subscribers': int(course_row['num_subscribers']),
#                 'num_reviews': int(course_row['num_reviews']),
#                 'num_lectures': int(course_row['num_lectures']),
#                 'level': str(course_row['level']),
#                 'content_duration': float(course_row['content_duration']),
#                 'published_timestamp': str(course_row['published_timestamp']),
#                 'subject': str(course_row['subject'])
#             })
        
#         return recommendations
    
#     except Exception as e:
#         logger.error(f"Error in recommendation: {str(e)}")
#         return [{"error": f"Error: {str(e)}"}]
def recommend_similar_courses(course_title, level=None, subject=None, data_file='Data/udemy_courses.csv', num_recommendations=20):
    """
    Recommend courses similar to the input course based on course_title, level, and subject.
    
    Parameters:
    - course_title (str): Title of the input course.
    - level (str, optional): Level of the input course (e.g., 'beginner', 'intermediate', 'advanced').
    - subject (str, optional): Subject/category of the input course.
    - data_file (str): Path to the CSV file containing course data.
    - num_recommendations (int): Number of courses to recommend.
    
    Returns:
    - list: List of dictionaries containing recommended course details.
    """
    try:
        # Load precomputed models
        with open('models/courses.pkl', 'rb') as f:
            df = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        logger.info(f"Loaded {len(df)} courses and precomputed TF-IDF matrix")
        
        # Clean data
        df['course_title'] = df['course_title'].fillna('')
        df['level'] = df['level'].str.lower().fillna('unknown')
        df['subject'] = df['subject'].str.lower().fillna('unknown')
        
        # Normalize input
        course_title = course_title.lower()
        level = level.lower() if level else None
        subject = subject.lower() if subject else None
        
        # Check if course_title exists in dataset
        input_course = df[df['course_title'].str.lower() == course_title]
        
        if not input_course.empty:
            # Course found in dataset
            input_course = input_course.iloc[0]
            input_level_value = get_level_value(input_course['level'])
            input_text = ' '.join([
                input_course['course_title'].lower(),
                input_course['level'].lower(),
                input_course['subject'].lower()
            ])
            logger.info(f"Course '{course_title}' found in dataset with level '{input_course['level']}' and subject '{input_course['subject']}'.")
        else:
            # Course not found, use input values or defaults
            if level and subject:
                logger.info(f"Course '{course_title}' not found in dataset. Using provided level '{level}' and subject '{subject}'.")
                input_level_value = get_level_value(level)
                input_text = ' '.join([
                    course_title,
                    level,
                    subject
                ])
            else:
                logger.info(f"Course '{course_title}' not found in dataset. Using default level 'beginner' and subject 'unknown'.")
                input_level_value = get_level_value('beginner')
                input_text = ' '.join([
                    course_title,
                    'beginner',
                    'unknown'
                ])
        
        # Compute input vector and similarity
        input_vector = vectorizer.transform([input_text])
        similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
        logger.info(f"Similarity scores range: {similarity_scores.min()} to {similarity_scores.max()}")
        
        # Create similarity DataFrame
        similarity_df = pd.DataFrame({
            'course_id': df['course_id'],
            'course_title': df['course_title'],
            'similarity': similarity_scores,
            'level': df['level'],
            'subject': df['subject'],
            'num_reviews': df['num_reviews'],
            'num_subscribers': df['num_subscribers']
        })
        
        # Filter candidates (exclude the input course if it exists in dataset)
        candidates = similarity_df[
            (similarity_df['course_title'].str.lower() != course_title) |
            ((level is not None) & (similarity_df['level'].str.lower() != level)) |
            ((subject is not None) & (similarity_df['subject'].str.lower() != subject))
        ]
        candidates = candidates[similarity_df['similarity'] > 0]
        logger.info(f"Number of candidates after filtering: {len(candidates)}")
        
        if candidates.empty:
            logger.warning("No candidates found after filtering.")
            return [{"warning": "No similar courses found due to filtering."}]
        
        # Apply level filter
        candidates = candidates.copy()
        candidates['level_value'] = candidates['level'].apply(get_level_value)
        candidates = candidates[candidates['level_value'] >= input_level_value]
        logger.info(f"Number of candidates after level filtering: {len(candidates)}")
        
        if candidates.empty:
            logger.warning("No candidates found after level filtering.")
            return [{"warning": "No courses found with level equal to or higher than the input course."}]
        
        # Sort by similarity and select top candidates
        top_candidates = candidates.sort_values(by='similarity', ascending=False).head(num_recommendations)
        logger.info(f"Top candidates: {top_candidates[['course_title', 'similarity', 'level']].to_dict('records')}")
        
        # Format output
        recommendations = []
        for _, row in top_candidates.iterrows():
            course_row = df[df['course_title'] == row['course_title']].iloc[0]
            recommendations.append({
                'course_id': str(course_row['course_id']),
                'course_title': str(course_row['course_title']),
                'url': 'https://primefaces.org/cdn/primeng/images/card-ng.jpg',
                'is_paid': bool(course_row['is_paid']),
                'price': str(course_row['price']),
                'course_rating': random.randint(1, 5),
                'num_subscribers': int(course_row['num_subscribers']),
                'num_reviews': int(course_row['num_reviews']),
                'num_lectures': int(course_row['num_lectures']),
                'level': str(course_row['level']),
                'content_duration': float(course_row['content_duration']),
                'published_timestamp': str(course_row['published_timestamp']),
                'category': str(course_row['subject'])
            })
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        return [{"error": f"Error: {str(e)}"}]

def get_popular_courses(courses_list, num_recommendations=5):
    """
    Get the most popular courses based on number of reviews and subscribers.
    
    Parameters:
    - courses_list (pd.DataFrame): DataFrame containing course data.
    - num_recommendations (int): Number of courses to return.
    
    Returns:
    - list: List of dictionaries containing popular course details.
    """
    try:
        df = courses_list.copy()
        df['num_reviews'] = pd.to_numeric(df['num_reviews'], errors='coerce').fillna(0)
        df['num_subscribers'] = pd.to_numeric(df['num_subscribers'], errors='coerce').fillna(0)
        
        # Normalize review and subscriber counts
        max_reviews = df['num_reviews'].max() or 1
        max_subscribers = df['num_subscribers'].max() or 1
        df['review_score'] = df['num_reviews'] / max_reviews
        df['subscriber_score'] = df['num_subscribers'] / max_subscribers
        
        # Compute popularity score
        df['popularity_score'] = 0.6 * df['review_score'] + 0.4 * df['subscriber_score']
        
        # Sort and select top courses
        top_courses = df.sort_values(by='popularity_score', ascending=False).head(num_recommendations)
        logger.info(f"Top popular courses: {top_courses[['course_title', 'popularity_score']].to_dict('records')}")
        
        # Format output
        recommendations = []
        for _, row in top_courses.iterrows():
            recommendations.append({
                'course_id': str(row['course_id']),
                'course_title': str(row['course_title']),
                'url': str(row['url']),
                'is_paid': bool(row['is_paid']),
                'price': str(row['price']),
                'num_subscribers': int(row['num_subscribers']),
                'num_reviews': int(row['num_reviews']),
                'num_lectures': int(row['num_lectures']),
                'level': str(row['level']),
                'content_duration': float(row['content_duration']),
                'published_timestamp': str(row['published_timestamp']),
                'subject': str(row['subject'])
            })
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error in popular courses: {str(e)}")
        return [{"error": f"Error: {str(e)}"}]

def recommend_collaborative(user_id, data_file='Data/udemy_courses.csv', num_recommendations=20):
    """
    Recommend courses for a user based on collaborative filtering using SVD.
    
    Parameters:
    - user_id (int): ID of the user to recommend courses for.
    - data_file (str): Path to the CSV file containing course data.
    - num_recommendations (int): Number of courses to recommend.
    
    Returns:
    - list: List of dictionaries containing recommended course details.
    """
    try:
        # Load precomputed models
        with open('models/courses.pkl', 'rb') as f:
            df = pickle.load(f)
        with open('models/svd_model.pkl', 'rb') as f:
            svd = pickle.load(f)
        with open('models/user_item_matrix.pkl', 'rb') as f:
            user_item_matrix = pickle.load(f)
        logger.info(f"Loaded SVD model and user-item matrix for {len(df)} courses")
        
        # Load user data from database simulation
        ratings = pd.read_csv('Data/ratings.csv')  # Simulated ratings from reviews table
        if str(user_id) not in ratings['user_id'].astype(str).values:
            logger.warning(f"User ID {user_id} not found in ratings.")
            return [{"warning": f"User ID {user_id} not found in ratings."}]
        
        # Get courses not yet rated by the user
        user_ratings = ratings[ratings['user_id'].astype(str) == str(user_id)]
        rated_courses = user_ratings['course_id'].astype(str).values
        all_courses = df['course_id'].astype(str).values
        unrated_courses = [cid for cid in all_courses if cid not in rated_courses]
        
        if not unrated_courses:
            logger.warning(f"No unrated courses available for user {user_id}.")
            return [{"warning": "No unrated courses available for recommendation."}]
        
        # Predict ratings for unrated courses
        predictions = []
        for course_id in unrated_courses:
            pred = svd.predict(str(user_id), str(course_id))
            predictions.append((course_id, pred.est))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = predictions[:num_recommendations]
        logger.info(f"Top predictions for user {user_id}: {[(cid, score) for cid, score in top_predictions]}")
        
        # Format output
        recommendations = []
        for course_id, _ in top_predictions:
            course_row = df[df['course_id'].astype(str) == str(course_id)].iloc[0]
            recommendations.append({
                'course_id': str(course_row['course_id']),
                'course_title': str(course_row['course_title']),
                'url': 'https://primefaces.org/cdn/primeng/images/card-ng.jpg',
                'is_paid': bool(course_row['is_paid']),
                'price': str(course_row['price']),
                'course_rating': random.randint(1, 5),
                'num_subscribers': int(course_row['num_subscribers']),
                'num_reviews': int(course_row['num_reviews']),
                'num_lectures': int(course_row['num_lectures']),
                'level': str(course_row['level']),
                'content_duration': float(course_row['content_duration']),
                'published_timestamp': str(course_row['published_timestamp']),
                'subject': str(course_row['subject'])
            })
        
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error in collaborative recommendation: {str(e)}")
        return [{"error": f"Error: {str(e)}"}]

def recommend_user_user_cf(user_id, ratings_file='Data/ratings.csv', courses_file='Data/udemy_courses.csv', num_recommendations=30):
    try:
        # Load data
        ratings = pd.read_csv(ratings_file)
        courses = pd.read_csv(courses_file)

        # Create user-item matrix
        user_item_matrix = ratings.pivot_table(index='user_id', columns='course_id', values='rating').fillna(0)

        # Check if user exists
        if user_id not in user_item_matrix.index:
            return [{"warning": f"User {user_id} not found in dataset."}]

        # Compute cosine similarity
        similarity = cosine_similarity(user_item_matrix)
        similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

        # Get top similar users
        # similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]  # exclude self
        similar_users = similarity_df[user_id][similarity_df[user_id] > 0.3].sort_values(ascending=False)[1:6]
        # Get courses those users rated highly (rating >= 3 or 4)
        similar_user_ids = similar_users.index
        similar_ratings = ratings[ratings['user_id'].isin(similar_user_ids)]
        high_rated = similar_ratings[similar_ratings['rating'] >= 4]

        # Remove courses the current user already rated
        user_rated = ratings[ratings['user_id'] == user_id]['course_id'].tolist()
        recommendable = high_rated[~high_rated['course_id'].isin(user_rated)]

        # Get top courses
        top_courses = (
            recommendable.groupby('course_id')
            .agg(score=('rating', 'mean'), count=('rating', 'count'))
            .sort_values(['score', 'count'], ascending=False)
            .head(num_recommendations)
            .reset_index()
        )

        # Merge with course details
        recommendations = pd.merge(top_courses, courses, on='course_id', how='left')

        result = []
        for _, row in recommendations.iterrows():
            result.append({
                'course_id': row['course_id'],
                'course_title': row['course_title'],
                'url': 'https://primefaces.org/cdn/primeng/images/card-ng.jpg',
                'course_rating': random.randint(1, 5),
                'is_paid': row['is_paid'],
                'price': row['price'],
                'num_subscribers': row['num_subscribers'],
                'num_reviews': row['num_reviews'],
                'num_lectures': row['num_lectures'],
                'level': row['level'],
                'content_duration': row['content_duration'],
                'published_timestamp': row['published_timestamp'],
                'subject': row['subject']
            })
        if len(result) < num_recommendations:
            user_highest = ratings[ratings['user_id'] == user_id].sort_values(by='rating', ascending=False).head(1)
            if not user_highest.empty:
                course_id = user_highest.iloc[0]['course_id']
                course_row = courses[courses['course_id'] == course_id]
                if not course_row.empty:
                    course_title = course_row.iloc[0]['course_title']
                    cbf_recs = recommend_similar_courses(course_title)
                    existing_ids = set([c['course_id'] for c in result])
                    for cbf_course in cbf_recs:
                        if cbf_course['course_id'] not in existing_ids:
                            result.append(cbf_course)
                            if len(result) >= num_recommendations:
                                break

        return result if result else [{"message": "No strong recommendations found."}]

    except Exception as e:
        return [{"error": str(e)}]



def update_model(data_file='Data/udemy_courses.csv'):
    """
    Preprocess the Udemy dataset and save the processed data for recommendations.
    
    Parameters:
    - data_file (str): Path to the CSV file containing course data.
    
    Returns:
    - bool: True if update is successful.
    """
    logger.info("Starting model update")
    if not os.path.exists('models'):
        os.makedirs('models')

    try:
        # Load and preprocess data
        df = pd.read_csv(data_file)
        required_columns = [
            'course_id', 'course_title', 'url', 'is_paid', 'price', 'num_subscribers',
            'num_reviews', 'num_lectures', 'level', 'content_duration', 'published_timestamp', 'subject'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing columns in {data_file}: {missing_columns}")
            raise Exception(f"Missing columns in {data_file}: {missing_columns}")
        
        df['course_title'] = df['course_title'].fillna('')
        df['level'] = df['level'].str.lower().fillna('unknown')
        df['subject'] = df['subject'].str.lower().fillna('unknown')
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0).astype(str)
        df['num_subscribers'] = pd.to_numeric(df['num_subscribers'], errors='coerce').fillna(0).astype(int)
        df['num_reviews'] = pd.to_numeric(df['num_reviews'], errors='coerce').fillna(0).astype(int)
        df['num_lectures'] = pd.to_numeric(df['num_lectures'], errors='coerce').fillna(0).astype(int)
        df['content_duration'] = pd.to_numeric(df['content_duration'], errors='coerce').fillna(0).astype(float)
        df['published_timestamp'] = df['published_timestamp'].fillna('unknown')
        df['course_id'] = df['course_id'].astype(str)
        
        courses_list = df[[
            'course_id', 'course_title', 'url', 'is_paid', 'price', 'num_subscribers',
            'num_reviews', 'num_lectures', 'level', 'content_duration', 'published_timestamp', 'subject'
        ]]
        
        # Compute TF-IDF matrix for CBF
        df['combined_text'] = (
            df['course_title'].str.lower() + ' ' +
            df['level'].str.lower() + ' ' +
            df['subject'].str.lower()
        )
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
        
        # Load ratings data for CF
        ratings = pd.read_csv('Data/ratings.csv')
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['user_id', 'course_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        # Train SVD model
        param_grid = {'n_factors': [10, 20, 50], 'n_epochs': [10, 20, 30]}
        gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
        gs.fit(data)
        print(gs.best_params['rmse'])
        svd = SVD(n_factors=gs.best_params['rmse']['n_factors'], n_epochs=gs.best_params['rmse']['n_epochs'], random_state=42)
        svd.fit(trainset)
        
        # Create user-item matrix
        user_item_matrix = ratings.pivot(index='user_id', columns='course_id', values='rating').fillna(0)
        
        # Save preprocessed data
        with open('models/courses.pkl', 'wb') as f:
            pickle.dump(courses_list, f)
        with open('models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open('models/tfidf_matrix.pkl', 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        with open('models/svd_model.pkl', 'wb') as f:
            pickle.dump(svd, f)
        with open('models/user_item_matrix.pkl', 'wb') as f:
            pickle.dump(user_item_matrix, f)
        logger.info("All files saved successfully")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in model update: {str(e)}")
        raise Exception(f"Error in model update: {str(e)}")