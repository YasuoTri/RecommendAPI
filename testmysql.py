import pymysql

try:
    conn = pymysql.connect(
        host='localhost',
        port=8200,  # Chỉ định port 8200
        user='root',
        password='123',  # Mật khẩu bạn xác nhận
        database='course_recommendation'
    )
    print("Connected to MySQL successfully!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")