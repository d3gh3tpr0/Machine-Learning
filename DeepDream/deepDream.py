from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
from scipy import ndimage
import numpy as np
import argparse
import cv2

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def preprocess(p):
    image = load_img(p)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def deprocess(image):
    image = image.reshape((image.shape[1], image.shape[2], 3))
    image /= 2.0
    image += 0.5
    image *= 255.0
    image = np.clip(image, 0, 255). astype("uint8")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def resize_image(image, size):
    resized = np.copy(image)
    resized = ndimage.zoom(resized, (1, float(size[0]) / resized.shape[1], float(size[1])/resized.shape[2], 1), order=1)

    return resized

def eval_loss_and_gradients(X):
    output = fetchLossGrads([X])
    (loss, G) = (output[0], output[1])
    return (loss, G)

def gradient_ascent(X, iters, alpha, maxLoss=-np.inf):
    # loop over our number of iterations
    for i in range(0, iters):
        # compute the loss and gradient
        (loss, G) = eval_loss_and_gradients(X)

        # if the loss is greater than the max loss, break from the
        # loop early to prevent strange effects
        if loss > maxLoss:
            break

        # take a step
        print("[INFO] Loss at {}: {}".format(i, loss))
        X += alpha * G
    # return the output of gradient ascent
    return X

LAYERS = {
    "mixed2": 2.0,
    "mixed3": 0.5,
}

NUM_OCTAVE = 3
OCTAVE_SCALE = 1.4
ALPHA = 0.001
NUM_ITER = 50
MAX_LOSS = 10.0

K.set_learning_phase(0) #since we won’t be training
                        #our network but instead constructing a feedback loop for our input image by running Inception in
                        #reverse

print("[INFO] loading inception network...")
model = InceptionV3(weights="imagenet", include_top=False)
dream = model.input

loss= K.variable(0.0)
layerMap = {layer.name: layer for layer in model.layers}

for layerName in LAYERS:

    x = layerMap[layerName].output
    coeff = LAYERS[layerName]
    scaling = K.prod(K.cast((K.shape(x)), "float32"))
    loss = loss + coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetchLossGrads = K.function([dream], outputs)

image = preprocess('result.png')
dims = image.shape[1:3]

octaveDims = [dims]
print(octaveDims)

for i in range(1, NUM_OCTAVE):
    size = [int(d / (OCTAVE_SCALE ** i)) for d in dims]
    octaveDims.append(size)

octaveDims = octaveDims[::-1]
orig = np.copy(image)
shrunk = resize_image(image, octaveDims[0])

for (o, size) in enumerate(octaveDims):
    print("[INFO] starting octave {}...".format(o))
    image = resize_image(image, size)

    image = gradient_ascent(image, iters=NUM_ITER, alpha=ALPHA, maxLoss= MAX_LOSS)

    upscaled = resize_image(shrunk, size)
    downscaled = resize_image(orig, size)

    lost = downscaled - upscaled
    image += lost

    shrunk = resize_image(orig, size)

#image = gradient_ascent(image, iters=NUM_ITER, alpha=ALPHA, maxLoss= MAX_LOSS)

image = deprocess(image)
cv2.imwrite('result3.png', image)
