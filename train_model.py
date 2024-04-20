import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load audio data
j_samples = []
non_j_samples = []

for file in os.listdir('samples/j'):
    with open(os.path.join('samples/j', file), 'rb') as f:
        j_samples.append(np.frombuffer(f.read(), dtype=np.int16))

for file in os.listdir('samples/non-j'):
    with open(os.path.join('samples/non-j', file), 'rb') as f:
        non_j_samples.append(np.frombuffer(f.read(), dtype=np.int16))

# Pad sequences to equal length
max_length = max(len(sample) for sample in j_samples + non_j_samples)
j_samples = pad_sequences(j_samples, maxlen=max_length, dtype='int16', padding='post')
non_j_samples = pad_sequences(non_j_samples, maxlen=max_length, dtype='int16', padding='post')

# Prepare labels
j_labels = np.ones(len(j_samples))
non_j_labels = np.zeros(len(non_j_samples))

# Combine data and labels
X = np.concatenate((j_samples, non_j_samples), axis=0)
y = np.concatenate((j_labels, non_j_labels), axis=0)

# Build the model
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(max_length, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(X.reshape(-1, max_length, 1), y, epochs=5, batch_size=32, validation_split=0.2, callbacks=[early_stop, checkpoint])

# Save the trained model
model.save('model.h5')