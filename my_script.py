from flask import Flask

# Tạo đối tượng Flask
app = Flask(__name__)

# Định nghĩa route cho trang chủ
@app.route('/')
def hello_world():
    return 'Hello, World!'  # Khi truy cập vào trang chủ, trả về "Hello, World!"

# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)  # Chạy ứng dụng ở chế độ debug
