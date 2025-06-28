import cv2
import numpy as np
from tensorflow.keras.models import load_model as keras_load

def load_model():
    return keras_load("digit_model.h5")  # Pretrained on Char74K/MNIST

def preprocess(cell_img):
    img = cv2.threshold(cell_img, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

def predict_digit(cell_img, model):
    processed = preprocess(cell_img)
    pred = model.predict(processed, verbose=0)
    digit = np.argmax(pred)
    return digit if np.max(pred) > 0.7 else 0  # Use threshold to filter empty cells
