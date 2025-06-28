from fastapi import FastAPI, Query, HTTPException
from typing import Optional
import pickle
from CourseRecommendationSystem import recommend_similar_courses, update_model, get_popular_courses, recommend_collaborative,recommend_user_user_cf
import pandas as pd
import logging
from fastapi import UploadFile, File
import shutil
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained models at startup
try:
    courses_list = pickle.load(open('models/courses.pkl', 'rb'))
    tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    tfidf_matrix = pickle.load(open('models/tfidf_matrix.pkl', 'rb'))
    svd_model = pickle.load(open('models/svd_model.pkl', 'rb'))
    user_item_matrix = pickle.load(open('models/user_item_matrix.pkl', 'rb'))
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Model file missing: {e}")

app = FastAPI()

@app.get("/recommend-similar")
def get_similar_courses(
    course_title: str = Query(..., description="Course title for content-based recommendations"),
    num_recommendations: int = Query(default=20, ge=1, le=30, description="Number of recommendations to return")
):
    """
    Get recommendations for courses similar to the input course title.
    Returns a list of courses with details: course_id, course_title, url, is_paid, price, num_subscribers,
    num_reviews, num_lectures, level, content_duration, published_timestamp, subject.
    """
    recommendations = recommend_similar_courses(course_title, num_recommendations=num_recommendations)
    return {"recommendations": recommendations}

@app.get("/popular-courses")
def get_popular_courses_endpoint(
    num_recommendations: int = Query(default=5, ge=1, le=30, description="Number of popular courses to return")
):
    """
    Get the most popular courses based on number of reviews and subscribers.
    Returns a list of courses with details: course_id, course_title, url, is_paid, price, num_subscribers,
    num_reviews, num_lectures, level, content_duration, published_timestamp, subject.
    """
    recommendations = get_popular_courses(courses_list, num_recommendations=num_recommendations)
    return {"recommendations": recommendations}

@app.get("/recommend-usercf")
def get_user_user_cf_recommendations(
    user_id: int = Query(..., description="User ID for User-User CF"),
    num_recommendations: int = Query(default=10, ge=1, le=30, description="Number of recommendations to return")
):
    """
    Get course recommendations for a user using User-User Collaborative Filtering.
    """
    recommendations = recommend_user_user_cf(user_id=user_id, num_recommendations=num_recommendations)
    return {"recommendations": recommendations}


@app.get("/recommend-collaborative")
def get_collaborative_recommendations(
    user_id: int = Query(..., description="User ID for collaborative filtering recommendations"),
    num_recommendations: int = Query(default=20, ge=1, le=30, description="Number of recommendations to return")
):
    """
    Get course recommendations for a user based on collaborative filtering.
    Returns a list of courses with details: course_id, course_title, url, is_paid, price, num_subscribers,
    num_reviews, num_lectures, level, content_duration, published_timestamp, subject.
    """
    recommendations = recommend_collaborative(user_id, num_recommendations=num_recommendations)
    return {"recommendations": recommendations}


@app.post("/recommend/update-model")
async def receive_csv_files(
    courses_file: UploadFile = File(...),
    enrollments_file: UploadFile = File(...)
):
    """
    Nhận 2 file CSV từ Laravel và lưu vào thư mục Data/
    Sau đó gọi update_model() để cập nhật hệ thống gợi ý.
    """
    try:
        # Tạo thư mục nếu chưa có
        os.makedirs('Data', exist_ok=True)

        # Lưu file courses.csv
        with open('Data/udemy_courses.csv', 'wb') as f:
            shutil.copyfileobj(courses_file.file, f)

        # Lưu file ratings.csv
        with open('Data/ratings.csv', 'wb') as f:
            shutil.copyfileobj(enrollments_file.file, f)

        # Gọi cập nhật model
        from CourseRecommendationSystem import update_model
        update_model()  # sẽ đọc lại Data/udemy_courses.csv và ratings.csv

        return {"message": "✔️ Model updated from CSV files successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi cập nhật mô hình: {str(e)}")

# @app.post("/update-model")
# def trigger_model_update():
#     """
#     Trigger model retraining and update saved models.
#     """
#     try:
#         update_model()
#         # Reload models after update
#         global courses_list, tfidf_vectorizer, tfidf_matrix, svd_model, user_item_matrix
#         courses_list = pickle.load(open('models/courses.pkl', 'rb'))
#         tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
#         tfidf_matrix = pickle.load(open('models/tfidf_matrix.pkl', 'rb'))
#         svd_model = pickle.load(open('models/svd_model.pkl', 'rb'))
#         user_item_matrix = pickle.load(open('models/user_item_matrix.pkl', 'rb'))
#         return {"status": "Model updated successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Model update failed: {str(e)}")