import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from dataset import *
from model import *

x, y, init_l, init_psi = load_dataset('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/handheld/data1/syn/vi1.csv')

#batch_size = 512
batch_size = 32

do_training = True

my_train_data_generator = train_data_generator(x, y, batch_size)

print('x[0, :]: ', x[0, :])
print('y[0, :]: ', y[0, :])

print('x[1, :]: ', x[1, :])
print('y[1, :]: ', y[1, :])

print('max delta_l: ', np.amax(y[:, 0]))
print('min delta_l: ', np.amin(y[:, 0]))
print('max delta_psi: ', np.amax(y[:, 1]))
print('min delta_psi: ', np.amin(y[:, 1]))

steps = np.ceil(float(x.shape[0]) / batch_size)

if do_training:
	model = create_model()

	model_checkpoint = ModelCheckpoint('bidirectional_lstm.hdf5', verbose=1)

	history = model.fit(x, y, epochs=100, verbose=1, callbacks=[model_checkpoint], shuffle=False)
	#history = model.fit_generator(my_train_data_generator, steps_per_epoch=steps, epochs=100, verbose=1, callbacks=[model_checkpoint])

	plt.plot(history.history['loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.show()
else:
	model = load_model('bidirectional_lstm.hdf5')

my_test_data_generator = test_data_generator(x, batch_size)

yhat = model.predict(x, verbose=1)
#yhat = model.predict_generator(my_test_data_generator, steps=steps, verbose=1)

print('yhat.shape: ', yhat.shape)

cur_l = init_l
cur_psi = init_psi
pred_l = []
pred_l.append(np.array(cur_l))

#for delta_l_psi in y:
for delta_l_psi in yhat:
    delta_l = delta_l_psi[0]
    delta_psi = delta_l_psi[1]
    cur_psi = cur_psi + delta_psi
    cur_l[0] = cur_l[0] + delta_l * np.cos(cur_psi)
    cur_l[1] = cur_l[1] + delta_l * np.sin(cur_psi)
    pred_l.append(np.array(cur_l))

np.savetxt('pred_positions.txt', pred_l, delimiter=' ')