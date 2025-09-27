import tensorflow as tf
import numpy as np
from PIL import Image

class DroneDetector:
    def __init__(self):
        self.model = tf.keras.models.load_model('drone_classifier.h5')
    
    def detect_drone(self, image_path):
        img = Image.open(image_path).resize((100, 100)).convert('RGB')
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        return prediction < 0.5  # True if drone detected