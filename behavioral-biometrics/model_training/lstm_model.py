
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

df = pd.read_csv('../dataset/user_behavior.csv')
X = df.drop('is_fraud', axis=1).values
y = df['is_fraud'].values

X = X.reshape((X.shape[0], 1, X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model = Sequential([
    LSTM(64, input_shape=(1, X.shape[2])),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16)

model.save('../backend/lstm_model.h5')
