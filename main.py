import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from typing import Tuple, List
import os


class LeafDiseaseDetector:
    def __init__(self, model_path: str, class_names: List[str]):
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names

    def preprocess_image(self, img_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found at: {img_path}")

        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0,1]
        return img_array

    def predict(self, img_array: np.ndarray) -> Tuple[str, float]:
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class = self.class_names[predicted_class_idx]

        return predicted_class, confidence

    def display_prediction(self, img_path: str, prediction: str, confidence: float):
        img = image.load_img(img_path)
        plt.imshow(img)
        plt.title(f"Prediction: {prediction}\nConfidence: {confidence:.2%}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # Using only one model
    MODEL_PATH = 'vgg19_leaf_disease_model.h5'
    # or
    # MODEL_PATH = 'densenet_leaf_disease_model.h5'

    CLASS_NAMES = [
        'Potato_Early_blight',
        'Potato_Healthy',
        'Potato_Late_blight',
        'Sugarcane_Healthy',
        'Sugarcane_RedRot',
        'Sugarcane_RedRust'
    ]

    try:
        # Initialize detector with single model
        detector = LeafDiseaseDetector(MODEL_PATH, CLASS_NAMES)

        # Replace with your image path
        TEST_IMAGE = '0.jpg'

        # Preprocess image
        processed_img = detector.preprocess_image(TEST_IMAGE)

        # Get prediction
        prediction, confidence = detector.predict(processed_img)

        # Print results
        print("Leaf Disease Detection Results:")
        print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")

        # Display the prediction
        detector.display_prediction(TEST_IMAGE, prediction, confidence)

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please ensure:")
        print(f"1. The model file exists at: {MODEL_PATH}")
        print(f"2. The test image exists at: {TEST_IMAGE}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")