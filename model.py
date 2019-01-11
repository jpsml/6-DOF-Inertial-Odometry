from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense
from keras.optimizers import Adam

def create_model(window_size = 200):

    model = Sequential()

    model.add(Bidirectional(LSTM(96, return_sequences=True), input_shape=(window_size, 6)))
    model.add(Bidirectional(LSTM(96)))
    model.add(Dropout(0.25))
    model.add(Dense(2))

    model.summary()

    model.compile(optimizer = Adam(0.0015), loss = 'mean_squared_error')

    return model