import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ===================== DATASET PATHS =====================
train_dir = "dataset/train"
val_dir = "dataset/val"

# ===================== IMAGE SETTINGS =====================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25   # ⬅️ Increased for 16-class learning

# ===================== DATA GENERATORS =====================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ===================== CLASS COUNT =====================
NUM_CLASSES = train_gen.num_classes
print("✅ Detected Classes:", NUM_CLASSES)
print("📌 Class Indices:", train_gen.class_indices)

# ===================== BASE MODEL =====================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Freeze pretrained layers

# ===================== FINAL MODEL =====================
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation="softmax")
])

# ===================== COMPILE =====================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===================== TRAIN =====================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ===================== SAVE MODEL =====================
model.save("model.h5")
print("✅ Model saved successfully as model.h5")
