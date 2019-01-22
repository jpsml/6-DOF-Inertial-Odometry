from keras.models import Sequential, Model
from keras.layers import Bidirectional, CuDNNLSTM, Dropout, Dense, Input, Layer
from keras.initializers import Constant
from keras.optimizers import Adam
from keras import backend as K

def weighted_squared_error_xyz(y_true, y_pred):
    # TODO: replace by regressed weight value
    s = 1.
    precision = K.exp(-s)
    return K.sum(precision * (y_true - y_pred) ** 2. + s, -1)


def weighted_squared_error_wpqr(y_true, y_pred):
    # TODO: replace by regressed weight value
    s = 1.
    precision = K.exp(-s)
    return K.sum(precision * (y_true - y_pred) ** 2. + s, -1)


# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)


def create_pred_model_6d_quat(window_size=200):
    inp = Input((window_size, 6), name='inp')
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(inp)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    y1_pred = Dense(3)(drop2)
    y2_pred = Dense(4)(drop2)

    model = Model(inp, [y1_pred, y2_pred])
    
    return model


def create_train_model_6d_quat(pred_model, window_size=200):
    inp = Input(shape=(window_size, 6), name='inp')
    y1_pred, y2_pred = pred_model(inp)
    y1_true = Input(shape=(3,), name='y1_true')
    y2_true = Input(shape=(4,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    train_model = Model([inp, y1_true, y2_true], out)
    train_model.summary()
    return train_model


def create_model_6d_rvec(window_size=200):
    input_gyro_acc = Input((window_size, 6))
    lstm1 = Bidirectional(CuDNNLSTM(128, return_sequences=True))(input_gyro_acc)    
    drop1 = Dropout(0.25)(lstm1)
    lstm2 = Bidirectional(CuDNNLSTM(128))(drop1)    
    drop2 = Dropout(0.25)(lstm2)    
    output_delta_rvec = Dense(3)(drop2)
    output_delta_tvec = Dense(3)(drop2)

    model = Model(inputs = input_gyro_acc, outputs = [output_delta_rvec, output_delta_tvec])
    model.summary()
    model.compile(optimizer = Adam(0.0001), loss = 'mean_squared_error')
    
    return model


def create_model_6d_quat(window_size=200):
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