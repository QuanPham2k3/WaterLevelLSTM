from flask import Flask, request, jsonify
from keras.api.models import load_model
import numpy as np
import pandas as pd
import pickle

from sklearn.discriminant_analysis import StandardScaler

app = Flask(__name__)

try:
    model = load_model('model/mo_hinhLSTM1.h5', compile=False)
except TypeError as e:
    print(f"Error loading model: {e}")
    

# Load các scaler đã huấn luyện từ file
with open('model/scaler_tan_chau.pkl', 'rb') as f:
    scaler_tan_chau = pickle.load(f)

with open('model/scaler_chau_doc.pkl', 'rb') as f:
    scaler_chau_doc = pickle.load(f)

# Tạo ra các chuỗi dữ liệu đầu vào cho mô hình 
def create_sequences(features, time_steps, forecast_horizon):
    X = []
    for i in range(len(features) - time_steps - forecast_horizon + 1):
        X.append(features[i:i + time_steps])
    return np.array(X)

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận dữ liệu từ form
    start_year = int(request.form.get('start_year'))  # Năm bắt đầu
    end_year = int(request.form.get('end_year'))      # Năm kết thúc

    # Đọc dữ liệu từ file data_train.csv
    df = pd.read_csv('data/data_predict.csv')

    # Lọc dữ liệu theo năm
    df['Date'] = pd.to_datetime(df['Date'])  # Đảm bảo cột 'date' là kiểu datetime
    df_year = df[(df['Date'].dt.year >= start_year) & (df['Date'].dt.year <= end_year)]

    if df_year.empty:
        return jsonify({"error": "No data available for the specified years."}), 404

    # Giả định rằng bạn có các cột cần thiết trong df_year
    features = df_year[["Tân Châu","Châu Đốc","Rạch Giá","Xẻo Rô","Sông Đốc","Gành Hào",
                    "Mỹ Thanh","Bến Trại","An Thuận","Binh Đại","Vàm Kênh","Vũng Tàu","kratie","Tháng","Mùa"]]
    
    # Lấy giá trị thực tế cho Tân Châu và Châu Đốc
    actual_tan_chau = df_year['Tân Châu'].values
    actual_chau_doc = df_year['Châu Đốc'].values
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Tạo các chuỗi dữ liệu đầu vào cho mô hình
    time_steps = 24  # 1 ngày với dữ liệu giờ
    forecast_horizon = 24  # Dự đoán 24 giờ liên tiếp
    X_input = create_sequences(features_scaled, time_steps, forecast_horizon)

    # Dự đoán trên dữ liệu đầu vào
    y_pred = model.predict(X_input)

    # Reshape để có được dạng (n_samples, forecast_horizon, 2)
    y_pred_reshaped = y_pred.reshape(-1, forecast_horizon, 2)

    # Lấy giá trị dự đoán cho Tân Châu và Châu Đốc
    y_pred_tan_chau = y_pred_reshaped[:, :, 0]  # Dự đoán Tân Châu
    y_pred_chau_doc = y_pred_reshaped[:, :, 1]  # Dự đoán Châu Đốc

    # Chọn giá trị dự đoán đầu tiên từ mỗi chuỗi
    y_pred_tan_chau_first = y_pred_tan_chau[:, 0]  # Giá trị đầu tiên cho Tân Châu
    y_pred_chau_doc_first = y_pred_chau_doc[:, 0]  # Giá trị đầu tiên cho Châu Đốc

    # Khôi phục giá trị dự đoán về giá trị gốc
    y_pred_tan_chau_original = scaler_tan_chau.inverse_transform(y_pred_tan_chau_first.reshape(-1, 1)).flatten()
    y_pred_chau_doc_original = scaler_chau_doc.inverse_transform(y_pred_chau_doc_first.reshape(-1, 1)).flatten()

    # Tạo response JSON
    response = {
        'predicted_tan_chau': y_pred_tan_chau_original.tolist(),
        'predicted_chau_doc': y_pred_chau_doc_original.tolist(),
        'actual_tan_chau': actual_tan_chau.tolist(),
        'actual_chau_doc': actual_chau_doc.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
