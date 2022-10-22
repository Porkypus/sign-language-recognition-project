import os
import numpy as np
from constants import DATA_PATH, sequence_length
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical  # covert stuff to one-hot encoding
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

sequences, labels = [], []
label_map = {label: num for num, label in enumerate(os.listdir("../data"))}
signs = []
for sign in os.listdir("../data"):
    signs.append(sign)
    for sequence in os.listdir(os.path.join("../data", sign)):
        window = []
        for frame_num in os.listdir(os.path.join("../data", sign, sequence)):
            res = np.load(
                os.path.join(DATA_PATH, sign, str(sequence), f"{frame_num}.npy")
            )
            window.append(res)
        sequences.append(window)
        labels.append(label_map[sign])

signs = np.array(signs)
X = np.array(sequences)
y = to_categorical(labels).astype(int)  # why do we do onehot encoding here?

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()  # look at sequential video mentionned at 1:38:00
model.add(
    LSTM(
        64,
        return_sequences=True,
        activation="relu",
        input_shape=(sequence_length, 1662),
    )
)
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(
    LSTM(64, return_sequences=False, activation="relu")
)  # set to false because we are not returning sequences as next layer is dense
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(signs.shape[0], activation="softmax"))  # look at argmax logic

model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
model.summary()
model.save("action.h5")
