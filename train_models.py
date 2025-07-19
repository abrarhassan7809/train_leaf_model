import tensorflow as tf
from tensorflow.keras.applications import VGG19, DenseNet121
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing
import os

# --- Config ---
num_classes = 6
batch_size = 32
num_epochs = 5
img_height = 224
img_width = 224
data_dir = 'CropSense_DataSet'
model_dir = Path("Models")
os.makedirs(model_dir, exist_ok=True)

# --- Enable GPU memory growth ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU(s) available")
        print("Using GPU:", tf.test.gpu_device_name())
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")


# --- Class Definition ---
class LeafDiseaseClassifier:
    def __init__(self, data_dir, img_height=224, img_width=224, batch_size=32, num_classes=6):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train_ds = None
        self.val_ds = None

    def prepare_data(self):
        """Load and preprocess the dataset using tf.data."""
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            label_mode='categorical'
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            label_mode='categorical'
        )

        normalization_layer = tf.keras.layers.Rescaling(1./255)
        self.train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
        self.val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

    def build_vgg19_model(self):
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(self.img_height, self.img_width, 3))
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_densenet_model(self):
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(self.img_height, self.img_width, 3))
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_custom_cnn_model(self):
        model = Sequential([
            tf.keras.layers.Input(shape=(self.img_height, self.img_width, 3)),
            tf.keras.layers.Rescaling(1./255),

            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, model, epochs=10, model_name='Model'):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(f"{model_name}_best.keras", save_best_only=True)
        ]
        history = model.fit(self.train_ds, validation_data=self.val_ds, epochs=epochs, callbacks=callbacks)
        self.plot_history(history, model_name)
        return history

    def plot_history(self, history, model_name):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title(f'{model_name} Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'{model_name} Loss')
        plt.legend()
        plt.savefig(f"{model_name}_training_plot.png")  # save plots
        plt.close()

    def save_model(self, model, filename):
        if not filename.endswith('.keras') and not filename.endswith('.h5'):
            filename += '.keras'

        save_path = os.path.join(model_dir, filename)
        model.save(save_path)
        print(f"✅ Model saved as {filename}")


# --- Worker Functions ---
def train_vgg19():
    print("\n--- Training VGG19 ---")
    classifier = LeafDiseaseClassifier(data_dir, img_height, img_width, batch_size, num_classes)
    classifier.prepare_data()
    model = classifier.build_vgg19_model()
    classifier.train_model(model, epochs=num_epochs, model_name='VGG19')
    classifier.save_model(model, "VGG19_Model")

def train_densenet():
    print("\n--- Training DenseNet121 ---")
    classifier = LeafDiseaseClassifier(data_dir, img_height, img_width, batch_size, num_classes)
    classifier.prepare_data()
    model = classifier.build_densenet_model()
    classifier.train_model(model, epochs=num_epochs, model_name='DenseNet121')
    classifier.save_model(model, "DenseNet121_Model")

def train_custom_cnn():
    print("\n--- Training Custom CNN ---")
    classifier = LeafDiseaseClassifier(data_dir, img_height, img_width, batch_size, num_classes)
    classifier.prepare_data()
    model = classifier.build_custom_cnn_model()
    classifier.train_model(model, epochs=num_epochs, model_name='CustomCNN')
    classifier.save_model(model, "CustomCNN_Model")

# --- Run in parallel ---
if __name__ == "__main__":
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found.")
        exit()

    # Start parallel training
    p1 = multiprocessing.Process(target=train_vgg19)
    p2 = multiprocessing.Process(target=train_densenet)
    p3 = multiprocessing.Process(target=train_custom_cnn)

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()

    print("✅ All models trained and saved successfully.")
