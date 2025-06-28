import pandas as pd
from sqlalchemy import create_engine
import pymysql
import warnings

# Tắt cảnh báo pandas
warnings.filterwarnings('ignore')

# Cấu hình kết nối MySQL (XAMPP)
DATABASE_URL = "mysql+pymysql://root:123@localhost:8200/course_recommendation9"
try:
    engine = create_engine(DATABASE_URL)
except Exception as e:
    raise Exception(f"Failed to connect to database: {e}")

# Hàm làm sạch dữ liệu
def clean_data(df):
    df = df.fillna({
        'Course Name': 'Unknown',
        'University': 'Unknown',
        'Difficulty Level': 'Unknown',
        'Course Rating': 0,
        'Course URL': '',
        'Course Description': '',
        'Skills': '',
        'final_cc_cname_DI': 'Unknown',
        'LoE_DI': 'Unknown',
        'YoB': 0,
        'gender': '',
        'registered': 0,
        'viewed': 0,
        'explored': 0,
        'certified': 0,
        'nevents': 0,
        'ndays_act': 0,
        'nplay_video': 0,
        'nchapters': 0,
        'nforum_posts': 0
    })
    return df

# Nhập dữ liệu từ Coursera.csv vào bảng courses
# def import_courses():
#     try:
#         courses = pd.read_csv('Data/Coursera.csv')
#         courses = clean_data(courses)
#         courses = courses.rename(columns={
#             'Course Name': 'course_name',
#             'University': 'university',
#             'Difficulty Level': 'difficulty_level',
#             'Course Rating': 'course_rating',
#             'Course URL': 'course_url',
#             'Course Description': 'course_description',
#             'Skills': 'skills'
#         })
#         courses.to_sql('courses', engine, if_exists='append', index=False)
#         print("Imported courses successfully")
#     except Exception as e:
#         print(f"Error importing courses: {e}")
def import_courses():
    try:
        # Read CSV
        courses = pd.read_csv('Data/Coursera.csv')
        courses = clean_data(courses)
        
        # Rename columns to match database schema
        courses = courses.rename(columns={
            'Course Name': 'course_name',
            'University': 'university',
            'Difficulty Level': 'difficulty_level',
            'Course Rating': 'course_rating',
            'Course URL': 'course_url',
            'Course Description': 'course_description',
            'Skills': 'skills'
        })

        # Remove duplicates in CSV based on course_name
        courses = courses.drop_duplicates(subset=['course_name'], keep='first')
        print(f"After deduplication, {len(courses)} unique courses to import")

        # Fetch existing course names from database
        existing_courses = pd.read_sql("SELECT course_name FROM courses", engine)
        existing_names = set(existing_courses['course_name'].str.lower())

        # Filter out courses that already exist (case-insensitive)
        courses_to_import = courses[~courses['course_name'].str.lower().isin(existing_names)]
        print(f"Found {len(courses_to_import)} new courses to import")
        if courses['course_name'].isnull().any() or (courses['course_name'] == 'Unknown').any():
            raise ValueError("Course Name cannot be null or 'Unknown'")
        elif not courses_to_import.empty:
            # Insert new courses
            courses_to_import.to_sql('courses', engine, if_exists='append', index=False)
            print(f"Imported {len(courses_to_import)} courses successfully")
        else:
            print("No new courses to import")

    except Exception as e:
        print(f"Error importing courses: {e}")
        raise

# Nhập dữ liệu người dùng từ Courseuserbehavior.csv vào bảng users
def import_users():
    try:
        behaviors = pd.read_csv('Data/Courseuserbehavior.csv')
        behaviors = clean_data(behaviors)
        # Chỉ lấy bản ghi duy nhất dựa trên userid_DI
        users = behaviors[['userid_DI', 'final_cc_cname_DI', 'LoE_DI', 'YoB', 'gender']].drop_duplicates(subset=['userid_DI'])
        users.to_sql('users', engine, if_exists='append', index=False)
        print("Imported users successfully")
    except Exception as e:
        print(f"Error importing users: {e}")

# Nhập tương tác từ Courseuserbehavior.csv vào bảng interactions
def import_interactions():
    try:
        behaviors = pd.read_csv('Data/Courseuserbehavior.csv')
        behaviors = clean_data(behaviors)
        users = pd.read_sql("SELECT id, userid_DI FROM users", engine)
        courses = pd.read_sql("SELECT id, course_name FROM courses", engine)
        
        user_id_map = dict(zip(users['userid_DI'], users['id']))
        # Giả sử course_id trong Courseuserbehavior.csv là tên khóa học hoặc định dạng khác
        # Cần ánh xạ dựa trên course_name
        course_id_map = dict(zip(courses['course_name'], courses['id']))
        
        interactions = behaviors[[
            'userid_DI', 'course_id', 'registered', 'viewed', 'explored', 'certified',
            'nevents', 'ndays_act', 'nplay_video', 'nchapters', 'nforum_posts'
        ]].copy()
        interactions['user_id'] = interactions['userid_DI'].map(user_id_map)
        # Ánh xạ course_id sang id trong bảng courses (giả sử course_id là course_name)
        interactions['course_id'] = interactions['course_id'].map(course_id_map)
        interactions = interactions.rename(columns={
            'registered': 'rating',
            'viewed': 'viewed',
            'explored': 'explored',
            'certified': 'certified',
            'nevents': 'nevents',
            'ndays_act': 'ndays_act',
            'nplay_video': 'nplay_video',
            'nchapters': 'nchapters',
            'nforum_posts': 'nforum_posts'
        })
        # Loại bỏ bản ghi không ánh xạ được user_id hoặc course_id
        interactions = interactions.dropna(subset=['user_id', 'course_id'])
        # Chỉ chèn các cột hợp lệ vào bảng interactions
        interactions = interactions[['user_id', 'course_id', 'rating', 'viewed', 'explored', 'certified', 
                                    'nevents', 'ndays_act', 'nplay_video', 'nchapters', 'nforum_posts']]
        interactions.to_sql('interactions', engine, if_exists='append', index=False)
        print("Imported interactions successfully")
    except Exception as e:
        print(f"Error importing interactions: {e}")

# Chạy import
if __name__ == "__main__":
    import_courses()
    import_users()
    import_interactions()