import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from dataset import *
from model import *

#def generate_tajectory(init_l, init_psi, y):
def generate_tajectory(init_l, init_psi, y_delta_l, y_delta_psi):
	cur_l = np.array(init_l)
	cur_psi = np.array(init_psi)
	pred_l = []
	pred_l.append(np.array(cur_l))

	#for delta_l_psi in y:
	#    delta_l = delta_l_psi[0]
	#    delta_psi = delta_l_psi[1]

	for [delta_l, delta_psi] in zip(y_delta_l, y_delta_psi):
	    cur_psi = cur_psi + delta_psi
	    cur_l[0] = cur_l[0] + delta_l * np.cos(cur_psi)
	    cur_l[1] = cur_l[1] + delta_l * np.sin(cur_psi)
	    pred_l.append(np.array(cur_l))

	return np.reshape(pred_l, (len(pred_l), 2))


np.random.seed(0)

#x, y, init_l, init_psi = load_dataset('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/handheld/data1/syn/vi1.csv')

x, [y_delta_l, y_delta_psi], init_l, init_psi = load_dataset('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/handheld/data1/syn/vi1.csv')

#x, y_delta_psi, init_l, init_psi = load_dataset('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/handheld/data1/syn/vi1.csv')
#y_delta_l = np.genfromtxt('y_delta_l.txt', delimiter=' ')
#y_delta_l = np.reshape(y_delta_l, (y_delta_l.shape[0], 1))

#y_delta_psi = y_delta_psi / np.pi

do_training = True

print('x[0, :]: ', x[0, :])
#print('y[0, :]: ', y[0, :])
print('y_delta_l[0, :]: ', y_delta_l[0, :])
print('y_delta_psi[0, :]: ', y_delta_psi[0, :])

print('x[1, :]: ', x[1, :])
#print('y[1, :]: ', y[1, :])
print('y_delta_l[1, :]: ', y_delta_l[1, :])
print('y_delta_psi[1, :]: ', y_delta_psi[1, :])

#max_delta_l = np.amax(y[:, 0])
#min_delta_l = np.amin(y[:, 0])
#max_delta_psi = np.amax(y[:, 1])
#min_delta_psi = np.amin(y[:, 1])

max_delta_l = np.amax(y_delta_l[:, 0])
min_delta_l = np.amin(y_delta_l[:, 0])
max_delta_psi = np.amax(y_delta_psi[:, 0])
min_delta_psi = np.amin(y_delta_psi[:, 0])

print('max_delta_l: ', max_delta_l)
print('min_delta_l: ', min_delta_l)
print('max_delta_psi: ', max_delta_psi)
print('min_delta_psi: ', min_delta_psi)

scale_factor = (max_delta_l - min_delta_l) / (max_delta_psi - min_delta_psi)
print('scale_factor: ', scale_factor)

if do_training:
	model = create_model()

	model_checkpoint = ModelCheckpoint('bidirectional_lstm.hdf5', monitor='loss', save_best_only=True, verbose=1)

	#history = model.fit(x, y, batch_size=1, epochs=100, verbose=1, callbacks=[model_checkpoint], shuffle=False)
	history = model.fit(x, [y_delta_l, y_delta_psi], epochs=100, verbose=1, callbacks=[model_checkpoint], shuffle=False)
	#history = model.fit(x, y_delta_psi, epochs=100, verbose=1, callbacks=[model_checkpoint], shuffle=False)

	plt.plot(history.history['loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.show()

model = load_model('bidirectional_lstm.hdf5')

[yhat_delta_l, yhat_delta_psi] = model.predict(x, batch_size=1, verbose=1)

#yhat_delta_psi = model.predict(x, batch_size=1, verbose=1)
#yhat_delta_l = y_delta_l

plt.figure()
plt.plot(y_delta_l)
plt.plot(yhat_delta_l)
plt.title('Delta L Pred vs Ground Truth')
plt.ylabel('Delta L (m)')
plt.xlabel('Time (0.1s)')
plt.legend(['Delta L Ground Truth', 'Delta L Pred'], loc='upper left')

plt.figure()
plt.plot(y_delta_psi)
plt.plot(yhat_delta_psi)
plt.title('Delta Psi Pred vs Ground Truth')
plt.ylabel('Delta Psi (rad)')
plt.xlabel('Time (0.1s)')
plt.legend(['Delta Psi Ground Truth', 'Delta Psi Pred'], loc='upper left')
plt.show()

#yhat_delta_psi = yhat_delta_psi * np.pi

print('init_l: ', init_l)
print('init_psi: ', init_psi)
gt_trajectory = generate_tajectory(init_l, init_psi, y_delta_l, y_delta_psi)

print('init_l: ', init_l)
print('init_psi: ', init_psi)
pred_trajectory = generate_tajectory(init_l, init_psi, yhat_delta_l, yhat_delta_psi)

print('init_l: ', init_l)
print('init_psi: ', init_psi)
pred_trajectory_only_l = generate_tajectory(init_l, init_psi, yhat_delta_l, y_delta_psi)

plt.figure()
plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1])
plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1])
#plt.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1])
#plt.plot(pred_trajectory[0:200, 0], pred_trajectory[0:200, 1])
plt.title('Trajectory Pred vs Ground Truth')
plt.ylabel('Y (m)')
plt.xlabel('X (m)')
plt.legend(['Trajectory Ground Truth', 'Trajectory Pred'], loc='upper left')

plt.figure()
plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1])
plt.plot(pred_trajectory_only_l[:, 0], pred_trajectory_only_l[:, 1])
#plt.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1])
#plt.plot(pred_trajectory_only_l[0:200, 0], pred_trajectory_only_l[0:200, 1])
plt.title('Trajectory Pred Only Delta L vs Ground Truth')
plt.ylabel('Y (m)')
plt.xlabel('X (m)')
plt.legend(['Trajectory Ground Truth', 'Trajectory Pred Only Delta L'], loc='upper left')
plt.show()

np.savetxt('pred_positions.txt', pred_trajectory, delimiter=' ')