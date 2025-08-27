import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.models import load_model

# ------------------------------
# DATA LOADING
# ------------------------------
def load_data(data_dir):
    """
    Load .npz keypoint files and prepare X, y arrays.
    Each keypoint sequence is padded/truncated to max_frames.
    """
    X, y = [], []
    files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    class_names = sorted({f.split("_")[0] for f in files})
    class_map = {cls: i for i, cls in enumerate(class_names)}

    for file in files:
        label = file.split("_")[0]
        data = np.load(os.path.join(data_dir, file), allow_pickle=True)
        keypoints = data["keypoints"]  # shape (frames, 33, 3)
        X.append(keypoints)
        y.append(class_map[label])

    X = np.array(X, dtype=object)
    y = np.array(y)
    return X, y, class_map

# ------------------------------
# SAMPLE PREPARATION
# ------------------------------
def prepare_sample_from_array(keypoints, max_frames=316):
    """
    Flatten, pad/truncate keypoints to max_frames for model input.
    """
    sample = keypoints.reshape(keypoints.shape[0], -1)  # (frames, 99)
    if sample.shape[0] < max_frames:
        pad = np.zeros((max_frames - sample.shape[0], sample.shape[1]))
        sample = np.vstack([sample, pad])
    else:
        sample = sample[:max_frames]
    return np.expand_dims(sample, 0)  # add batch dimension

def prepare_sample(npz_path, max_frames=316):
    data = np.load(npz_path)["keypoints"]
    return prepare_sample_from_array(data, max_frames)

# ------------------------------
# MODEL BUILDING
# ------------------------------
def build_tcn(input_shape, num_classes):
    """
    Simple TCN-like model for sequence classification.
    """
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------
# TRAINING
# ------------------------------
if __name__ == "__main__":
    data_dir = "data/processed"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    X_raw, y, class_map = load_data(data_dir)
    print("Classes:", class_map)

    # Pad/truncate sequences
    max_frames = 316
    X = np.array([prepare_sample_from_array(seq, max_frames)[0] for seq in X_raw])
    print("X shape:", X.shape, "y shape:", y.shape)

    model = build_tcn((max_frames, X.shape[2]), len(class_map))
    model.summary()

    # Train
    model.fit(X, y, epochs=10, batch_size=2, validation_split=0.2)

    # Save
    model_path = os.path.join(model_dir, "shot_classifier.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Quick test
    sample_path = os.path.join(data_dir, "coverdrive_cover1.npz")
    sample = prepare_sample(sample_path, max_frames)
    pred = model.predict(sample)
    print("Sample prediction:", pred)
