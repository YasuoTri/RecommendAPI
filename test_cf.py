import pytest
import pandas as pd
import time
from fastapi.testclient import TestClient
from main import app
from CourseRecommendationSystem import recommend_collaborative
import pickle
import logging

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
client = TestClient(app)

@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    df = pd.read_csv('Data/udemy_courses.csv')
    return df

@pytest.fixture
def ratings_data():
    """Load ratings data for testing."""
    df = pd.read_csv('Data/ratings.csv')
    return df

@pytest.fixture
def courses_list():
    """Load precomputed courses list."""
    with open('models/courses.pkl', 'rb') as f:
        return pickle.load(f)

def test_recommendation_exists(ratings_data):
    """Test if recommendations are returned for an existing user via API."""
    user_id = ratings_data['user_id'].iloc[0]
    response = client.get(f"/recommend-collaborative?user_id={user_id}")
    assert response.status_code == 200
    recommendations = response.json()['recommendations']
    assert isinstance(recommendations, list), "Recommendations must be a list"
    assert len(recommendations) > 0, "No recommendations returned"
    assert all('course_id' in rec for rec in recommendations), "Each recommendation must have course_id"
    rated_courses = ratings_data[ratings_data['user_id'] == user_id]['course_id'].astype(str).values
    assert all(rec['course_id'] not in rated_courses for rec in recommendations), "Recommendations should not include rated courses"

def test_recommendation_nonexistent():
    """Test if error is returned for a nonexistent user via API."""
    response = client.get("/recommend-collaborative?user_id=99999")
    assert response.status_code == 200
    recommendations = response.json()['recommendations']
    assert isinstance(recommendations, list), "Response must be a list"
    assert len(recommendations) == 1, "Should return exactly one error message"
    assert "warning" in recommendations[0], "Response must contain warning key"
    assert "not found" in recommendations[0]["warning"].lower(), "Warning message must indicate user not found"

def test_precision_at_k(ratings_data, sample_data, k=5, num_samples=3):
    """
    Test precision@k to ensure recommendations are relevant based on user similarity.
    Calculates average precision@k across a sample of users.
    """
    user_ids = ratings_data['user_id'].unique()[:num_samples]
    precisions = []
    
    for user_id in user_ids:
        user_ratings = ratings_data[ratings_data['user_id'] == user_id]
        user_subjects = sample_data[sample_data['course_id'].isin(user_ratings['course_id'])]['subject'].str.lower().unique()
        
        recommendations = recommend_collaborative(user_id, num_recommendations=k)
        if isinstance(recommendations, list) and recommendations and ("error" in recommendations[0] or "warning" in recommendations[0]):
            logger.info(f"Recommendations failed for user {user_id}: {recommendations[0]}")
            continue
        
        relevant = 0
        for rec in recommendations:
            rec_subject = rec['subject'].lower().strip()
            if rec_subject in user_subjects:
                relevant += 1
        
        precision = relevant / k
        precisions.append(precision)
        logger.info(f"Precision@{k} for user {user_id}: {precision:.3f}")
    
    if not precisions:
        pytest.fail("No valid recommendations found for any tested user")
    
    avg_precision = sum(precisions) / len(precisions)
    logger.info(f"Average Precision@{k} across {len(precisions)} users: {avg_precision:.3f}")
    assert avg_precision >= 0.6, f"Average Precision@{k} too low: {avg_precision:.3f} (expected >= 0.6)"

def test_runtime_performance(ratings_data):
    """Test runtime performance for a single recommendation via API."""
    user_id = ratings_data['user_id'].iloc[0]
    start_time = time.time()
    response = client.get(f"/recommend-collaborative?user_id={user_id}&num_recommendations=20")
    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"Runtime for 20 recommendations: {runtime:.3f} seconds")
    assert runtime < 0.5, f"Runtime too slow: {runtime:.3f} seconds (expected < 0.5)"
    assert response.status_code == 200

def test_recommendation_format(ratings_data):
    """Test if recommendations follow the expected format via API."""
    user_id = ratings_data['user_id'].iloc[0]
    response = client.get(f"/recommend-collaborative?user_id={user_id}")
    recommendations = response.json()['recommendations']
    if isinstance(recommendations, list) and recommendations and ("error" in recommendations[0] or "warning" in recommendations[0]):
        pytest.skip(f"Recommendations failed for user {user_id}: {recommendations[0]}")
    
    expected_keys = [
        'course_id', 'course_title', 'url', 'is_paid', 'price', 'num_subscribers',
        'num_reviews', 'num_lectures', 'level', 'content_duration', 'published_timestamp', 'subject'
    ]
    for rec in recommendations:
        assert all(key in rec for key in expected_keys), f"Missing keys in recommendation: {rec}"
        assert isinstance(rec['course_id'], str), f"course_id must be str: {rec['course_id']}"
        assert isinstance(rec['course_title'], str), f"course_title must be str: {rec['course_title']}"
        assert isinstance(rec['url'], str), f"url must be str: {rec['url']}"
        assert isinstance(rec['is_paid'], bool), f"is_paid must be bool: {rec['is_paid']}"
        assert isinstance(rec['price'], str), f"price must be str: {rec['price']}"
        assert isinstance(rec['num_subscribers'], int), f"num_subscribers must be int: {rec['num_subscribers']}"
        assert isinstance(rec['num_reviews'], int), f"num_reviews must be int: {rec['num_reviews']}"
        assert isinstance(rec['num_lectures'], int), f"num_lectures must be int: {rec['num_lectures']}"
        assert isinstance(rec['level'], str), f"level must be str: {rec['level']}"
        assert isinstance(rec['content_duration'], float), f"content_duration must be float: {rec['content_duration']}"
        assert isinstance(rec['published_timestamp'], str), f"published_timestamp must be str: {rec['published_timestamp']}"
        assert isinstance(rec['subject'], str), f"subject must be str: {rec['subject']}"

def test_user_similarity(ratings_data, sample_data):
    """Test if users A and B get similar recommendations, while C gets different ones."""
    user_a = 1  # User A
    user_b = 2  # User B
    user_c = 3  # User C
    
    response_a = client.get(f"/recommend-collaborative?user_id={user_a}&num_recommendations=5")
    response_b = client.get(f"/recommend-collaborative?user_id={user_b}&num_recommendations=5")
    response_c = client.get(f"/recommend-collaborative?user_id={user_c}&num_recommendations=5")
    
    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert response_c.status_code == 200
    
    recs_a = response_a.json()['recommendations']
    recs_b = response_b.json()['recommendations']
    recs_c = response_c.json()['recommendations']
    
    # Get course IDs
    course_ids_a = {rec['course_id'] for rec in recs_a}
    course_ids_b = {rec['course_id'] for rec in recs_b}
    course_ids_c = {rec['course_id'] for rec in recs_c}
    
    # Check overlap between A and B
    common_ab = course_ids_a.intersection(course_ids_b)
    logger.info(f"Common courses between A and B: {len(common_ab)}")
    assert len(common_ab) >= 2, f"Expected at least 2 common courses between A and B, got {len(common_ab)}"
    
    # Check C is distinct
    common_ac = course_ids_a.intersection(course_ids_c)
    common_bc = course_ids_b.intersection(course_ids_c)
    logger.info(f"Common courses between A and C: {len(common_ac)}, B and C: {len(common_bc)}")
    assert len(common_ac) == 0, f"Expected no common courses between A and C, got {len(common_ac)}"
    assert len(common_bc) == 0, f"Expected no common courses between B and C, got {len(common_bc)}"