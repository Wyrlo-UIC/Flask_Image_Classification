import os
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

IMG_SIZE = (128, 128)
BATCH_SIZE = 8

def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(dataset_path, model_path):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training'
    )
    val_data = datagen.flow_from_directory(
        dataset_path, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation'
    )

    num_classes = len(train_data.class_indices)
    if num_classes < 2:
        return {"error": "At least 2 different classes with images are required."}

    model = create_model(num_classes)

    # Train the model
    history = model.fit(train_data, validation_data=val_data, epochs=50)

    # Save model and class indices
    model.save(model_path)
    class_indices_path = model_path.replace(".h5", "_class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(train_data.class_indices, f)

    # Get the final validation accuracy
    final_accuracy = history.history['val_accuracy'][-1]  # Last validation accuracy

    return {"accuracy": float(final_accuracy), "classes": list(train_data.class_indices.keys())}


def predict_image(image, model_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load class indices
    class_indices_path = model_path.replace(".h5", "_class_indices.json")
    if not os.path.exists(class_indices_path):
        return "Error: No trained model found. Please train the model first."

    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)

    class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping

    # Preprocess image
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image)
    class_index = np.argmax(predictions)

    return class_labels.get(class_index, "Unknown")