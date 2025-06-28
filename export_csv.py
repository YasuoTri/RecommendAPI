import pandas as pd
import pymysql
import numpy as np
import os

# Tạo thư mục nếu chưa có
os.makedirs('Data', exist_ok=True)

# Kết nối MySQL (XAMPP)
conn = pymysql.connect(
    host='localhost',
    port=8200,
    user='root',
    password='123',
    database='course_recommendation',
    charset='utf8mb4'
)

# ✅ Xuất Coursera_new.csv từ bảng `courses`, bao gồm instructor_id
courses_query = """
SELECT 
    c.id AS course_id,
    c.course_name AS 'Course Name',
    c.university AS 'University',
    c.difficulty_level AS 'Difficulty Level',
    c.course_rating AS 'Course Rating',
    c.course_url AS 'Course URL',
    c.course_description AS 'Course Description',
    c.skills AS 'Skills',
    c.price AS 'Price',
    c.status AS 'Status',
    GROUP_CONCAT(cat.name) AS 'Categories',
    GROUP_CONCAT(ci.instructor_id) AS 'Instructor IDs'
FROM courses c
LEFT JOIN course_category cc ON c.id = cc.course_id
LEFT JOIN categories cat ON cc.category_id = cat.id
LEFT JOIN course_instructors ci ON c.id = ci.course_id
GROUP BY c.id
"""
courses_df = pd.read_sql(courses_query, conn)
# Handle NULL instructor IDs
courses_df['Instructor IDs'] = courses_df['Instructor IDs'].fillna('0')
courses_df.to_csv('Data/Coursera_new.csv', index=False, encoding='utf-8-sig')
print("✅ Exported Data/Coursera_new.csv")

# ✅ Xuất Courseuserbehavior_new.csv từ `interactions`, `users`
# user_behavior_query = """
# SELECT 
#     i.course_id,
#     u.userid_DI,
#     i.viewed,
#     i.explored,
#     i.start_time AS start_time_DI,
#     i.last_event AS last_event_DI,
#     i.nevents,
#     i.ndays_act,
#     i.nplay_video,
#     i.nforum_posts
# FROM interactions i
# JOIN users u ON i.user_id = u.id
# """
# user_behavior_df = pd.read_sql(user_behavior_query, conn)
# user_behavior_query = """
# SELECT 
#     i.course_id,
#     u.userid_DI,
#     i.viewed,
#     i.explored,
#     i.start_time AS start_time_DI,
#     i.last_event AS last_event_DI,
#     i.nevents,
#     i.ndays_act,
#     i.nplay_video,
#     (SELECT COUNT(*) FROM forum_posts fp WHERE fp.user_id = i.user_id AND fp.course_id = i.course_id) AS nforum_posts,
#     (SELECT IF(COUNT(*) > 0, 1, 0) FROM certificates c WHERE c.user_id = i.user_id AND c.course_id = i.course_id) AS certified,
#     (SELECT COUNT(DISTINCT lp.lesson_id) FROM lesson_progress lp JOIN lessons l ON lp.lesson_id = l.id WHERE lp.user_id = i.user_id AND l.course_id = i.course_id AND lp.status IN ('in_progress', 'completed')) AS nchapters,
#     (SELECT AVG(qr.score) FROM quiz_results qr JOIN quizzes q ON qr.quiz_id = q.id JOIN lessons l ON q.lesson_id = l.id WHERE qr.user_id = i.user_id AND l.course_id = i.course_id) AS avg_quiz_score,
#     r.rating,
#     u.LoE_DI,
#     u.YoB,
#     u.gender
# FROM interactions i
# JOIN users u ON i.user_id = u.id
# LEFT JOIN reviews r ON i.user_id = r.user_id AND i.course_id = r.course_id
# """
# user_behavior_df = pd.read_sql(user_behavior_query, conn)
# # Gán giá trị mặc định nếu rating là NULL
# user_behavior_df['rating'] = user_behavior_df['rating'].fillna(0)

# # Thêm các cột mô phỏng tương tự edX format
# user_behavior_df['index'] = range(len(user_behavior_df))
# user_behavior_df['Random'] = np.random.randint(1, 1000, size=len(user_behavior_df))
# # user_behavior_df['registered'] = 1
# # user_behavior_df['grade'] = np.random.uniform(0, 1, size=len(user_behavior_df)).round(2)
# # user_behavior_df['incomplete_flag'] = np.random.choice([0, 1], size=len(user_behavior_df), p=[0.8, 0.2])

# # Chuyển đổi datetime
# user_behavior_df['start_time_DI'] = pd.to_datetime(user_behavior_df['start_time_DI'], errors='coerce')
# user_behavior_df['last_event_DI'] = pd.to_datetime(user_behavior_df['last_event_DI'], errors='coerce')

# # Đảm bảo thứ tự cột
# # columns = [
# #     'index', 'Random', 'course_id', 'userid_DI', 'registered',
# #     'viewed', 'explored', 'certified',
# #     'final_cc_cname_DI', 'LoE_DI', 'YoB', 'gender',
# #     'grade', 'start_time_DI', 'last_event_DI',
# #     'nevents', 'ndays_act', 'nplay_video', 'nchapters',
# #     'nforum_posts', 'roles', 'incomplete_flag'
# # ]
# columns = [
#     'index', 'course_id', 'userid_DI',
#     'viewed', 'explored', 'rating',
#     'LoE_DI', 'YoB', 'gender',
#     'start_time_DI', 'last_event_DI',
#     'nevents', 'ndays_act', 'nplay_video',
#     'nforum_posts', 'certified', 'nchapters', 'avg_quiz_score'
# ]
# user_behavior_df = user_behavior_df[columns]

# # Lưu ra CSV
# user_behavior_df.to_csv('Data/Courseuserbehavior_new.csv', index=False, encoding='utf-8-sig')
# print("✅ Exported Data/Courseuserbehavior_new.csv")
user_behavior_query = """
SELECT 
    i.course_id,
    u.userid_DI,
    i.viewed,
    i.explored,
    i.start_time AS start_time_DI,
    i.last_event AS last_event_DI,
    i.nevents,
    i.ndays_act,
    i.nplay_video,
    (SELECT COUNT(*) FROM forum_posts fp WHERE fp.user_id = i.user_id AND fp.course_id = i.course_id) AS nforum_posts,
    (SELECT IF(COUNT(*) > 0, 1, 0) FROM certificates c WHERE c.user_id = i.user_id AND c.course_id = i.course_id) AS certified,
    (SELECT COUNT(DISTINCT lp.lesson_id) FROM lesson_progress lp JOIN lessons l ON lp.lesson_id = l.id WHERE lp.user_id = i.user_id AND l.course_id = i.course_id AND lp.status IN ('in_progress', 'completed')) AS nchapters,
    (SELECT AVG(qr.score) FROM quiz_results qr JOIN quizzes q ON qr.quiz_id = q.id JOIN lessons l ON q.lesson_id = l.id WHERE qr.user_id = i.user_id AND l.course_id = i.course_id) AS avg_quiz_score,
    r.rating,
    u.LoE_DI,
    u.YoB,
    u.gender
FROM interactions i
JOIN users u ON i.user_id = u.id
LEFT JOIN reviews r ON i.user_id = r.user_id AND i.course_id = r.course_id
"""
user_behavior_df = pd.read_sql(user_behavior_query, conn)
# Gán giá trị mặc định nếu rating hoặc avg_quiz_score là NULL
user_behavior_df['rating'] = user_behavior_df['rating']
user_behavior_df['avg_quiz_score'] = user_behavior_df['avg_quiz_score']

# Thêm cột index
user_behavior_df['index'] = range(len(user_behavior_df))

# Chuyển đổi datetime
user_behavior_df['start_time_DI'] = pd.to_datetime(user_behavior_df['start_time_DI'], errors='coerce')
user_behavior_df['last_event_DI'] = pd.to_datetime(user_behavior_df['last_event_DI'], errors='coerce')

# Đảm bảo thứ tự cột
columns = [
    'index', 'course_id', 'userid_DI',
    'viewed', 'explored', 'rating',
    'LoE_DI', 'YoB', 'gender',
    'start_time_DI', 'last_event_DI',
    'nevents', 'ndays_act', 'nplay_video',
    'nforum_posts', 'certified', 'nchapters',
    'avg_quiz_score'
]
user_behavior_df = user_behavior_df[columns]

# Lưu ra CSV
user_behavior_df.to_csv('Data/Courseuserbehavior_new.csv', index=False, encoding='utf-8-sig')
print("✅ Exported Data/Courseuserbehavior_new.csv")

# ✅ Xuất quiz_results.csv từ bảng `quiz_results`
quiz_results_query = """
SELECT 
    user_id,
    quiz_id,
    score,
    started_at,
    completed_at
FROM quiz_results
"""
quiz_results_df = pd.read_sql(quiz_results_query, conn)
quiz_results_df.to_csv('Data/quiz_results.csv', index=False, encoding='utf-8-sig')
print("✅ Exported Data/quiz_results.csv")

# ✅ Xuất enrollments.csv từ bảng `enrollments`
enrollments_query = """
SELECT 
    user_id,
    course_id,
    enrolled_at,
    completed_at,
    status
FROM enrollments
"""
enrollments_df = pd.read_sql(enrollments_query, conn)
enrollments_df.to_csv('Data/enrollments.csv', index=False, encoding='utf-8-sig')
print("✅ Exported Data/enrollments.csv")

reviews_query = """
SELECT 
    id,
    user_id,
    course_id,
    rating,
    comment,
    feedback_type,
    created_at,
    deleted_at,
    updated_at
FROM reviews
"""
reviews_df = pd.read_sql(reviews_query, conn)
reviews_df.to_csv('Data/reviews.csv', index=False, encoding='utf-8-sig')
print("✅ Exported Data/reviews.csv")

# ✅ Xuất student_categories.csv từ bảng `student_category`
student_categories_query = """
SELECT 
    sc.student_id,
    s.user_id,
    sc.category_id,
    cat.name AS category_name
FROM student_category sc
JOIN categories cat ON sc.category_id = cat.id
JOIN students s ON sc.student_id = s.id
"""
student_categories_df = pd.read_sql(student_categories_query, conn)
student_categories_df.to_csv('Data/student_categories.csv', index=False, encoding='utf-8-sig')
print("✅ Exported Data/student_categories.csv")
# ✅ Đóng kết nối
conn.close()