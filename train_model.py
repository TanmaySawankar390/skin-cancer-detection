import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Add this line
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import json
import numpy as np

# Dataset parameters
IMG_SIZE = (300, 300)  # Increased size for B3
BATCH_SIZE = 32
EPOCHS_INITIAL = 30
EPOCHS_FINE_TUNE = 20

# Data augmentation
def create_datagen():
    return ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

# Load and prepare data
def load_data():
    train_datagen = create_datagen()
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        './Train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_gen = test_datagen.flow_from_directory(
        './Test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    return train_gen, test_gen

# Build advanced model
def build_model(num_classes):
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1536, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(768, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.4)(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

# Learning rate schedule
def lr_schedule(epoch):
    initial_lr = 1e-3
    decay_factor = 0.9
    return initial_lr * (decay_factor ** (epoch // 2))

# Training pipeline
def train():
    train_gen, test_gen = load_data()
    num_classes = len(train_gen.class_indices)
    
    # Save class indices
    with open("class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f)

    # Build model
    model, base_model = build_model(num_classes)
    
    # Initial training
    for layer in base_model.layers:
        layer.trainable = False
        
    model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])

    callbacks = [
        ModelCheckpoint("best_init_model.keras", save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3),
        LearningRateScheduler(lr_schedule)
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS_INITIAL,
        validation_data=test_gen,
        callbacks=callbacks
    )

    # Fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:200]:
        layer.trainable = False

    model.compile(optimizer=AdamW(1e-5, weight_decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])

    history_fine = model.fit(
        train_gen,
        epochs=EPOCHS_INITIAL + EPOCHS_FINE_TUNE,
        initial_epoch=history.epoch[-1],
        validation_data=test_gen,
        callbacks=callbacks
    )

    model.save("skin_cancer_model.keras")
    print("✅ Training complete")

if __name__ == "__main__":
    train()