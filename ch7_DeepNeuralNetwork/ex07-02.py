
# 심층 신경망
# 2개의 은닉층

import keras
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기

# 기존 코드 (Fashion MNIST)
# (train_input, train_target), (test_input, test_target) = \
#     keras.datasets.fashion_mnist.load_data()

# 수정 코드 (MNIST 손글씨 숫자)
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.mnist.load_data()


# 2. 데이터 전처리

train_scaled = train_input / 255.0

# 기존 코드 (1차원으로 변환)
# train_scaled = train_scaled.reshape(-1, 28*28)

# 수정 코드
# Flatten 층에서 1차원으로 변환하므로 reshape 생략


train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)


# 3. 기존 심층 신경망 코드 (주석 처리)

# inputs = keras.layers.Input(shape=(784,))
# dense1 = keras.layers.Dense(100, activation='sigmoid')
# dense2 = keras.layers.Dense(10, activation='softmax')

# model = keras.Sequential([inputs, dense1, dense2])
# model.summary()


# 4. 층을 추가하는 다른 방법 (기존 코드)

# model = keras.Sequential([
#     keras.layers.Input(shape=(784,)),
#     keras.layers.Dense(100, activation='sigmoid', name='은닉층'),
#     keras.layers.Dense(10, activation='softmax', name='출력층')
# ], name='패션 MNIST 모델')

# model.summary()


# 기존 add() 방식
# model = keras.Sequential()
# model.add(keras.layers.Input(shape=(784,)))
# model.add(keras.layers.Dense(100, activation='sigmoid'))
# model.add(keras.layers.Dense(10, activation='softmax'))

# model.summary()

# model.compile(loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_scaled, train_target, epochs=5)


# 5. 렐루 함수 (기존 코드)

# model = keras.Sequential()
# model.add(keras.layers.Input(shape=(28,28)))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(100, activation='relu'))
# model.add(keras.layers.Dense(10, activation='softmax'))

# model.summary()

# model.compile(loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_scaled, train_target, epochs=5)
# model.evaluate(val_scaled, val_target)


# 6. 수정된 심층 신경망 (은닉층 2개)

model = keras.Sequential()

model.add(keras.layers.Input(shape=(28,28)))

model.add(keras.layers.Flatten())

# 기존 은닉층 (1개)
# model.add(keras.layers.Dense(100, activation='relu'))

# 수정 은닉층 (2개)
model.add(keras.layers.Dense(100, activation='relu', name='hidden1'))
model.add(keras.layers.Dense(50, activation='relu', name='hidden2'))

# 출력층
model.add(keras.layers.Dense(10, activation='softmax', name='output'))

model.summary()


# 7. 옵티마이저

# 기존 코드
# model.compile(loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# 수정 코드 (Adam 옵티마이저)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# 8. 모델 훈련

model.fit(train_scaled, train_target, epochs=5)


# 9. 모델 평가

model.evaluate(val_scaled, val_target)

