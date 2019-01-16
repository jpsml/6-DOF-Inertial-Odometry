from keras.models import Sequential, Model
from keras.layers import Bidirectional, CuDNNLSTM, Dropout, Dense, Input
from keras.optimizers import Adam

def create_model(window_size=200):

    #model = Sequential()
    #model.add(CuDNNLSTM(128, input_shape=(window_size, 6)))
    ##model.add(Bidirectional(CuDNNLSTM(96, return_sequences=True), input_shape=(window_size, 6)))
    ##model.add(Dropout(0.25))
    ##model.add(Bidirectional(CuDNNLSTM(96)))
    ##model.add(Dropout(0.25))
    #model.add(Dense(2))
    ##model.add(Dense(1))
    #model.summary()
    ##model.compile(optimizer = Adam(0.0015), loss = 'mean_squared_error')
    #model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')

    input_gyro_acc = Input((window_size, 6))
    #lstm1 = CuDNNLSTM(128)(input_gyro_acc)
    #lstm1 = Bidirectional(CuDNNLSTM(96, return_sequences=True))(input_gyro_acc)
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)
    # TODO: add dropout layer
    #lstm2 = Bidirectional(CuDNNLSTM(96))(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(lstm1)
    # TODO: add dropout layer
    #output_delta_l = Dense(1)(lstm1)
    #output_delta_psi = Dense(1)(lstm1)
    output_delta_l = Dense(1)(lstm2)
    output_delta_psi = Dense(1)(lstm2)
    model = Model(inputs = input_gyro_acc, outputs = [output_delta_l, output_delta_psi])
    model.summary()
    #model.compile(optimizer = Adam(0.0015), loss = 'mean_squared_error', loss_weights = [1.0, 0.030225446880354216])
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model