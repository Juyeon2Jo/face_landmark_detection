import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

# CSV 파일 경로
csv_file_path = 'face_landmarks.csv'

# CSV 파일 로드
data = pd.read_csv(csv_file_path)

# 첫 번째 열(이미지 이름 등)이 NaN인 행을 제거
data.dropna(subset=[data.columns[0]], inplace=True)

# 정규화
scaler = StandardScaler()

# X, Y, Z 좌표 분리
x_columns = data.columns[1:469]  # 첫 번째 열은 이미지 이름이므로 제외
y_columns = data.columns[469:937]
z_columns = data.columns[937:1405]

x_data = data[x_columns]
y_data = data[y_columns]
z_data = data[z_columns]

# 정규화 적용
x_scaled = scaler.fit_transform(x_data)
y_scaled = scaler.fit_transform(y_data)
z_scaled = scaler.fit_transform(z_data)

# 각 좌표별로 파일 저장
with open('x_principal_components.pkl', 'wb') as file:
    pickle.dump(x_scaled, file)

with open('y_principal_components.pkl', 'wb') as file:
    pickle.dump(y_scaled, file)

with open('z_principal_components.pkl', 'wb') as file:
    pickle.dump(z_scaled, file)
