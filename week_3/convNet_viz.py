import matplotlib.pyplot as plt

f, axarr = plt.subplots(3,4)

FIRST_IMAGE = 0
SECOND_IMAGE = 7
THIRD_IMAGE = 26
CONVOLUTION_NUMBER = 1

from tensorflow.keras import models

layer_outputs = [layer.output for layer in model.layers]
activiation_model = tf.keras.models.Model(input = model.input, outputs = layers_outputs)

for x in range(0, 4):
    f1 = activiation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f1[0, : , :, CONVOLUTION_NUMBER ], cmap='plasma')
    axarr[0, x].grid(False)

    f2 = activiation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f2[0, : , :, CONVOLUTION_NUMBER ], cmap='plasma')
    axarr[0, x].grid(False)

    f3 = activiation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f3[0, : , :, CONVOLUTION_NUMBER ], cmap='plasma')
    axarr[0, x].grid(False)


