import keras
(train_input, train_target), (test_input, test_target) =\
        keras.datasets.fashion_mnist.load_data()

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1,10,figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

import numpy as np

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1,28*28)

print(train_scaled.shape)

#from sklearn.model_selection import cross_validate
#from sklearn.linear_model import SGDClassifier

#sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
#scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
#print(np.mean(scores['test_score']))

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

inputs = keras.layers.Input(shape=(784,))
dense1 = keras.layers.Dense(100, activation='sigmoid')
dense2 = keras.layers.Dense(10, activation='softmax')

model = keras.Sequential([inputs, dense1, dense2])

model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(train_target[:10])

model.fit(train_scaled, train_target, epochs=5)

model.evaluate(val_scaled, val_target)

model.summary()
