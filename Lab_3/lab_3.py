import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

data_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
data_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

X = data_train.iloc[:, 1:].to_numpy().astype("float32") / 255.0
y = data_train.iloc[:, 0].to_numpy()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=123
)

y_train_oh = keras.utils.to_categorical(y_train, num_classes=10)
y_valid_oh = keras.utils.to_categorical(y_valid, num_classes=10)

nn_model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

nn_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = nn_model.fit(
    X_train, y_train_oh,
    validation_data=(X_valid, y_valid_oh),
    epochs=10,
    batch_size=32,
    verbose=1
)

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Епохи")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Динаміка точності")
plt.show()

val_loss, val_acc = nn_model.evaluate(X_valid, y_valid_oh, verbose=0)
print(f"Точність на валідації: {val_acc:.4f}")

pred_probs = nn_model.predict(X_valid)
y_pred = np.argmax(pred_probs, axis=1)

cmatrix = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cmatrix, annot=True, fmt="d", cmap="crest")
plt.xlabel("Прогноз")
plt.ylabel("Істинні значення")
plt.title("Confusion Matrix")
plt.show()


def visualize_samples(images, true_labels, pred_labels, n_samples=6):
    chosen = random.sample(range(len(images)), n_samples)
    plt.figure(figsize=(12, 4))
    for idx, sample in enumerate(chosen):
        plt.subplot(1, n_samples, idx + 1)
        plt.imshow(images[sample].reshape(28, 28), cmap="gray")
        plt.title(f"Передбачено: {pred_labels[sample]}\nПравильно: {true_labels[sample]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


visualize_samples(X_valid, y_valid, y_pred, n_samples=5)
