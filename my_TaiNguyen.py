import matplotlib
matplotlib.use('Agg')  # Sử dụng backend Agg để không cần GUI

import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, send_file, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import io

app = Flask(__name__)

# Ánh xạ các mức độ ưu tiên thành giá trị số
priority_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

@app.route('/endpoint', methods=['POST'])
def process_data():
    # Lấy dữ liệu JSON từ request
    data = request.get_json()

    # In ra dữ liệu nhận được
    print("Dữ liệu nhận được từ Spring Boot:", data)

    # Lấy thông tin dự án (projectID)
    project_id = data.get("id", "N/A")  # Nếu không có projectID trong request, sẽ mặc định là "N/A"

    # Xử lý và tạo DataFrame từ dữ liệu trên
    team_members_data = []

    # Lặp qua tất cả các thành viên
    for member in data["teamMembers"]:
        total_salary = 0
        avg_finish = 0
        avg_priority = 0
        avg_days_to_complete = 0
        num_issues = len(member["issues"])

        # Lặp qua các nhiệm vụ của từng thành viên
        for issue in member["issues"]:
            # Chuyển đổi salary từ chuỗi sang số (float)
            total_salary += float(issue["salary"])

            # Xử lý priority: chuyển 'Medium', 'Low', 'High' thành số
            priority_value = priority_mapping.get(issue["priority"], 0)  # Mặc định là 0 nếu không có trong ánh xạ
            avg_priority += priority_value

            # Chuyển finish và priority về kiểu số nguyên
            avg_finish += int(issue["finish"])

            # Tính số ngày hoàn thành nhiệm vụ
            due_date = datetime.strptime(f"{issue['dueDate'][0]}-{issue['dueDate'][1]}-{issue['dueDate'][2]}", "%Y-%m-%d")
            actual_date = datetime.strptime(f"{issue['actualDate'][0]}-{issue['actualDate'][1]}-{issue['actualDate'][2]}", "%Y-%m-%d")
            days_to_complete = abs((due_date - actual_date).days)
            avg_days_to_complete += days_to_complete
        
        # Tính các giá trị trung bình
        avg_finish /= num_issues
        avg_priority /= num_issues
        avg_days_to_complete /= num_issues
        
        # Thêm thông tin của thành viên vào danh sách
        team_members_data.append({
            "teamMemberID": member["id"],
            "fullname": member["fullname"],
            "totalSalary": total_salary,
            "avgFinish": avg_finish,
            "avgPriority": avg_priority,
            "avgDaysToComplete": avg_days_to_complete,
            "numIssues": num_issues,  # Thêm trường numIssues
            "issues": member["issues"]
        })

    # Chuyển danh sách thành pandas DataFrame
    df_member = pd.DataFrame(team_members_data)

    # Tính hiệu suất trung bình của tất cả thành viên
    average_performance = df_member['avgFinish'].mean()
    print(f"Hiệu suất trung bình của dự án: {average_performance:.2f}")

    # Tính hiệu suất theo tháng
    df_member['actualMonth'] = df_member['issues'].apply(lambda issues: [datetime.strptime(f"{issue['actualDate'][0]}-{issue['actualDate'][1]}-{issue['actualDate'][2]}", "%Y-%m-%d").strftime('%Y-%m') for issue in issues])

    # Chuyển đổi thành dataframe mới với 'actualMonth' và 'avgFinish'
    df_monthly = df_member.explode('actualMonth')

    # Nhóm theo tháng và tính trung bình hiệu suất cho tất cả thành viên trong tháng đó
    monthly_performance = df_monthly.groupby('actualMonth')['avgFinish'].mean().to_dict()
    print(f"Hiệu suất theo tháng: {monthly_performance}")

    # Các đặc trưng và mục tiêu (target)
    X = df_member[['totalSalary', 'avgPriority', 'avgDaysToComplete']]  # Các đặc trưng
    y = df_member['avgFinish']  # Mục tiêu (mức độ hoàn thành trung bình)

    # Kiểm tra số lượng mẫu dữ liệu
    if len(X) > 1:
        # Chia dữ liệu thành bộ huấn luyện và bộ kiểm tra chỉ khi số mẫu > 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Khởi tạo mô hình RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Huấn luyện mô hình
        model.fit(X_train, y_train)

        # Dự đoán hiệu suất trên bộ kiểm tra
        y_pred = model.predict(X_test)

        # Đánh giá mô hình bằng Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse:.2f}')
    else:
        print("Không đủ dữ liệu để chia thành bộ huấn luyện và kiểm tra.")
        # Sử dụng mô hình hoặc phương pháp khác nếu chỉ có một mẫu dữ liệu
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = "N/A"
        print("Dự đoán hiệu suất (do chỉ có một mẫu dữ liệu):", y_pred) 


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Huấn luyện lại mô hình với dữ liệu đã chuẩn hóa
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    # Dự đoán hiệu suất cho tất cả thành viên dựa trên dữ liệu đã chuẩn hóa
    df_member['predictedFinish'] = model.predict(scaler.transform(X))

    # Trả về kết quả dự đoán
    result_data = []
    for index, row in df_member.iterrows():
        result_data.append({
            "teamMemberID": row['teamMemberID'],
            "fullname": row['fullname'],
            "totalSalary": row['totalSalary'],
            "avgFinish": row['avgFinish'],
            "avgPriority": row['avgPriority'],
            "avgDaysToComplete": row['avgDaysToComplete'],
            "numIssues": row['numIssues'],  # Thêm trường numIssues
            "predictedFinish": row['predictedFinish']
        })

    return jsonify({
        "projectID": project_id,
        "meanSquaredError": mse,
        "teamMembersData": result_data,
        "averagePerformance": average_performance,
        "monthlyPerformance": monthly_performance
    })


if __name__ == '__main__':
    app.run(debug=True)
