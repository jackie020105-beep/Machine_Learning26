import keras

model = keras.models.load_model('best-cnn-model.keras')
print(model.layers)


conv = model.layers[0]
conv_weights = conv.weights[0].numpy()

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,16,figsize=(15,2))

for i in range(2):
    for j in range(16):
        axs[i,j].imshow(conv_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

#conv = model.layers[2]
#conv_weights = conv.weights[0].numpy()

#import matplotlib.pyplot as plt
#fig, axs = plt.subplots(4,16,figsize=(15,4))

#for i in range(4):
#    for j in range(16):
#        axs[i,j].imshow(conv_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
#        axs[i, j].axis('off')
#plt.show()

conv_acti = keras.Model(model.inputs[0], model.layers[0].output)

import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
plt.imshow(train_input[0], cmap='gray_r')
plt.show()

ankle_boot = train_input[0:1].reshape(-1, 28, 28, 1)/255.0
feature_maps = conv_acti.predict(ankle_boot)

print(feature_maps.shape)

fig, axs = plt.subplots(4, 8, figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')
plt.show()

conv2_acti = keras.Model(model.inputs, model.layers[2].output)
feature_maps = conv2_acti.predict(ankle_boot)
print(feature_maps.shape)

fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8 + j])
        axs[i, j].axis('off')
plt.show()
