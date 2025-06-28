import pytest
import pandas as pd
import time
from fastapi.testclient import TestClient
from main import app
from CourseRecommendationSystem import recommend_similar_courses
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
client = TestClient(app)

@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    df = pd.read_csv('Data/udemy_courses.csv')
    return df

@pytest.fixture
def courses_list():
    """Load precomputed courses list."""
    with open('models/courses.pkl', 'rb') as f:
        return pickle.load(f)

def test_recommendation_exists(sample_data):
    """Test if recommendations are returned for an existing course via API."""
    course_title = sample_data['course_title'].iloc[0]
    response = client.get(f"/recommend-similar?course_title={course_title}")
    assert response.status_code == 200
    recommendations = response.json()['recommendations']
    assert isinstance(recommendations, list), "Recommendations must be a list"
    assert len(recommendations) > 0, "No recommendations returned"
    assert all('course_id' in rec for rec in recommendations), "Each recommendation must have course_id"
    assert all(rec['course_title'] != course_title for rec in recommendations), "Recommendations should not include input course"

def test_recommendation_nonexistent():
    """Test if error is returned for a nonexistent course via API."""
    response = client.get("/recommend-similar?course_title=Nonexistent Course")
    assert response.status_code == 200
    recommendations = response.json()['recommendations']
    assert isinstance(recommendations, list), "Response must be a list"
    assert len(recommendations) == 1, "Should return exactly one error message"
    assert "error" in recommendations[0], "Response must contain error key"
    assert "not found" in recommendations[0]["error"].lower(), "Error message must indicate course not found"

def test_precision_at_k(sample_data, k=5, num_samples=3):
    """
    Test precision@k to ensure recommendations are relevant based on subject and level.
    Calculates average precision@k across a sample of courses.
    """
    course_titles = sample_data['course_title'].unique()[:num_samples]
    precisions = []
    
    for course_title in course_titles:
        course_row = sample_data[sample_data['course_title'] == course_title].iloc[0]
        input_subject = course_row['subject'].lower().strip()
        input_level_value = {'beginner level': 1, 'intermediate level': 2, 'expert level': 3, 'all levels': 4, 'unknown': 0}.get(course_row['level'].lower().strip(), 0)
        
        recommendations = recommend_similar_courses(course_title, num_recommendations=k)
        if isinstance(recommendations, list) and recommendations and ("error" in recommendations[0] or "warning" in recommendations[0]):
            logger.info(f"Recommendations failed for course {course_title}: {recommendations[0]}")
            continue
        
        relevant = 0
        for rec in recommendations:
            rec_subject = rec['subject'].lower().strip()
            rec_level = rec['level'].lower().strip()
            rec_level_value = {'beginner level': 1, 'intermediate level': 2, 'expert level': 3, 'all levels': 4, 'unknown': 0}.get(rec_level, 0)
            if rec_subject == input_subject and rec_level_value >= input_level_value:
                relevant += 1
        
        precision = relevant / k
        precisions.append(precision)
        logger.info(f"Precision@{k} for course {course_title}: {precision:.3f}")
    
    if not precisions:
        pytest.fail("No valid recommendations found for any tested course")
    
    avg_precision = sum(precisions) / len(precisions)
    logger.info(f"Average Precision@{k} across {len(precisions)} courses: {avg_precision:.3f}")
    assert avg_precision >= 0.6, f"Average Precision@{k} too low: {avg_precision:.3f} (expected >= 0.6)"

def test_runtime_performance(sample_data):
    """Test runtime performance for a single recommendation via API."""
    course_title = sample_data['course_title'].iloc[0]
    start_time = time.time()
    response = client.get(f"/recommend-similar?course_title={course_title}&num_recommendations=20")
    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"Runtime for 20 recommendations: {runtime:.3f} seconds")
    assert runtime < 0.5, f"Runtime too slow: {runtime:.3f} seconds (expected < 0.5)"
    assert response.status_code == 200

def test_recommendation_format(sample_data):
    """Test if recommendations follow the expected format via API."""
    course_title = sample_data['course_title'].iloc[0]
    response = client.get(f"/recommend-similar?course_title={course_title}")
    recommendations = response.json()['recommendations']
    if isinstance(recommendations, list) and recommendations and ("error" in recommendations[0] or "warning" in recommendations[0]):
        pytest.skip(f"Recommendations failed for course {course_title}: {recommendations[0]}")
    
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

def test_level_filter(sample_data):
    """Test if recommendations respect the level hierarchy (equal or higher than input course level)."""
    course_title = sample_data[sample_data['level'].str.lower() == 'beginner level']['course_title'].iloc[0]
    response = client.get(f"/recommend-similar?course_title={course_title}&num_recommendations=5")
    recommendations = response.json()['recommendations']
    
    if isinstance(recommendations, list) and recommendations and ("error" in recommendations[0] or "warning" in recommendations[0]):
        pytest.skip(f"Recommendations failed for course {course_title}: {recommendations[0]}")
    
    input_level_value = 1  # Beginner Level
    for rec in recommendations:
        rec_level = rec['level'].lower().strip()
        rec_level_value = {'beginner level': 1, 'intermediate level': 2, 'expert level': 3, 'all levels': 4, 'unknown': 0}.get(rec_level, 0)
        assert rec_level_value >= input_level_value, f"Recommended course {rec['course_title']} has lower level ({rec_level}) than input course (Beginner Level)"