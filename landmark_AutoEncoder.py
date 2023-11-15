from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from keras.models import Model
import tensorflow as tf
import numpy as np
import pickle

# x, y, z 데이터 불러오기
with open('x_principal_components.pkl', 'rb') as file:
    x_data = pickle.load(file)

with open('y_principal_components.pkl', 'rb') as file:
    y_data = pickle.load(file)

with open('z_principal_components.pkl', 'rb') as file:
    z_data = pickle.load(file)
x_train = np.stack((x_data, y_data, z_data), axis=-1)

print(x_train.shape)  # 형태 확인

input_dim = (468, 3, 1)
encoder_input = Input(shape=input_dim)

# 인코더 구조
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
encoded = Dense(128, activation='relu')(x)

# 디코더 구조
x = Dense(468 * 3 * 1)(encoded)  # 중간 차원을 크게 설정
x = Reshape((468, 3, 1))(x)  # 3차원 형태로 변환

# Conv2DTranspose 레이어를 조정하여 원하는 출력 크기를 얻습니다.
x = Conv2DTranspose(16, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)  # 스트라이드 조정
x = Conv2DTranspose(32, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)  # 스트라이드 조정
decoder_output = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)  # 채널 수를 3으로 변경

auto_encoder = Model(encoder_input, decoder_output)
auto_encoder.compile(optimizer='adam', loss='mean_squared_error')

# x_train 형태 조정
x_train = np.expand_dims(x_train, axis=-1)

# 훈련 설정
batch_size = 32
epochs = 50
validation_split = 0.2  # 검증 데이터로 사용할 비율 (예: 20%)

# 모델 훈련
auto_encoder.fit(
    x_train, x_train,  # 입력 데이터와 레이블(여기서는 동일)
    batch_size=batch_size,
    epochs=epochs,
    validation_split=validation_split
)

