from keras.layers import Input, Dense
from keras.models import Model
import pickle

input_dim = 468 * 3  # 입력 차원
# 변환된 데이터 불러오기
with open('principal_components.pkl', 'rb') as file:
    X_train = pickle.load(file)

# Autoencoder 모델 구성
input_layer = Input(shape=(input_dim,))
encoded = Dense(256, activation='relu')(input_layer)
encoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
autoencoder = Model(input_layer, decoded)

# 컴파일
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 모델 학습
autoencoder.fit(X_train, X_train,  # 입력과 출력이 동일
                epochs=50,        # 에폭 수
                batch_size=256,   # 배치 크기
                shuffle=True,     # 데이터 셔플
                validation_split=0.2)  # 검증 데이터 비율
