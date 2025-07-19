import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing import image_dataset_from_directory
from typing import Tuple, List
from tensorflow.keras.preprocessing import image


class LeafDiseaseDetector:
    def __init__(self, model_path: str, class_names: List[str]):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_size = (224, 224)

    def preprocess_image(self, img_path: str) -> np.ndarray:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found at: {img_path}")
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0

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

    def evaluate_model(self, test_dir: str):
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found at: {test_dir}")

        print("[INFO] Loading test dataset...")
        test_ds = image_dataset_from_directory(
            test_dir,
            image_size=self.img_size,
            batch_size=32,
            shuffle=False
        )

        print("[INFO] Predicting on test dataset...")
        y_true = []
        y_pred = []

        for images, labels in test_ds:
            predictions = self.model.predict(images)
            predicted_labels = tf.argmax(predictions, axis=1).numpy()
            true_labels = labels.numpy()
            y_true.extend(true_labels)
            y_pred.extend(predicted_labels)

        print("\n[INFO] Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        print("[INFO] Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    MODEL_PATH = 'Models/VGG19_Model.keras'
    TEST_IMAGE = 'CropSense_DataSet/Sugarcane_RedRust/0.jpg'
    TEST_FOLDER = 'CropSense_DataSet'

    CLASS_NAMES = [
        'Potato_Early_blight', 'Potato_Healthy', 'Potato_Late_blight', 'Sugarcane_Healthy',
        'Sugarcane_RedRot', 'Sugarcane_RedRust'
    ]

    try:
        detector = LeafDiseaseDetector(MODEL_PATH, CLASS_NAMES)

        # Single image prediction
        processed_img = detector.preprocess_image(TEST_IMAGE)
        prediction, confidence = detector.predict(processed_img)
        print(f"[SINGLE IMAGE] Prediction: {prediction} (Confidence: {confidence:.2%})")
        detector.display_prediction(TEST_IMAGE, prediction, confidence)

        # Full model evaluation
        print("\n[MODEL EVALUATION] Starting...")
        detector.evaluate_model(TEST_FOLDER)

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error during prediction or evaluation: {str(e)}")
