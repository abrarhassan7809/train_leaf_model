from tensorflow.keras.applications import VGG19, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, training will use CPU")


class LeafDiseaseClassifier:
    def __init__(self, data_dir, img_height=224, img_width=224, batch_size=32, num_classes=6):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train_generator = None
        self.val_generator = None

    def prepare_data(self):
        """Load and augment the dataset using ImageDataGenerator."""
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        self.val_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

    def build_vgg19_model(self):
        """Build and compile a VGG19-based model."""
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(self.img_height, self.img_width, 3))
        base_model.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def build_densenet_model(self):
        """Build and compile a DenseNet121-based model."""
        base_model = DenseNet121(weights='imagenet', include_top=False,
                                 input_shape=(self.img_height, self.img_width, 3))
        base_model.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train_model(self, model, epochs=10, model_name='Model'):
        """Train the model and plot training history."""
        history = model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs
        )
        self.plot_history(history, model_name)
        return history

    def plot_history(self, history, model_name):
        """Plot training and validation accuracy/loss."""
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_name} Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Loss')
        plt.legend()
        plt.show()

    def save_model(self, model, filename):
        """Save the trained model to a file."""
        model.save(filename)
        print(f"Model saved as {filename}")


if __name__ == "__main__":
    # Initialize the classifier
    classifier = LeafDiseaseClassifier(data_dir='CropSense_DataSet')

    # Prepare the data
    classifier.prepare_data()

    # Build models
    vgg19_model = classifier.build_vgg19_model()
    densenet_model = classifier.build_densenet_model()

    # Train models
    print("Training VGG19...")
    vgg_history = classifier.train_model(vgg19_model, epochs=5, model_name='VGG19')

    print("Training DenseNet121...")
    densenet_history = classifier.train_model(densenet_model, epochs=10, model_name='DenseNet121')

    # Save models
    classifier.save_model(vgg19_model, 'vgg19_leaf_disease_model.h5')
    classifier.save_model(densenet_model, 'densenet_leaf_disease_model.h5')