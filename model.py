from keras.models import Sequential, Model
from keras.layers import Bidirectional, CuDNNLSTM, Dropout, Dense, Input
from keras.optimizers import Adam

def create_model_6d(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_p = Dense(3)(drop2)
    output_delta_q = Dense(4)(drop2)

    model = Model(inputs = input_gyro_acc, outputs = [output_delta_p, output_delta_q])
    model.summary()
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model


def create_model_3d(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_l = Dense(1)(drop2)
    output_delta_theta = Dense(1)(drop2)
    output_delta_psi = Dense(1)(drop2)

    model = Model(inputs = input_gyro_acc, outputs = [output_delta_l, output_delta_theta, output_delta_psi])
    model.summary()
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model


def create_model_2d(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_l = Dense(1)(drop2)
    output_delta_psi = Dense(1)(drop2)
    model = Model(inputs = input_gyro_acc, outputs = [output_delta_l, output_delta_psi])
    model.summary()
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model