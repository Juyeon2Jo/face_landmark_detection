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
scaled_data = scaler.fit_transform(data)

# 결과 출력
print(data.head())
# 결과 확인
print(f"Original number of features: {data.shape[1]}")
print(f"Reduced number of features: {scaled_data.shape[1]}")

# 변환된 주성분 데이터 저장
with open('principal_components.pkl', 'wb') as file:
    pickle.dump(scaled_data, file)
