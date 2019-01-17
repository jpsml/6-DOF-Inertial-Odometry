import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from sklearn.utils import shuffle

from dataset import *
from model import *

def generate_trajectory(init_l, init_psi, y_delta_l, y_delta_psi):
	cur_l = np.array(init_l)
	cur_psi = np.array(init_psi)
	pred_l = []
	pred_l.append(np.array(cur_l))

	for [delta_l, delta_psi] in zip(y_delta_l, y_delta_psi):
	    cur_psi = cur_psi + delta_psi
	    cur_l[0] = cur_l[0] + delta_l * np.cos(cur_psi)
	    cur_l[1] = cur_l[1] + delta_l * np.sin(cur_psi)
	    pred_l.append(np.array(cur_l))

	return np.reshape(pred_l, (len(pred_l), 2))


np.random.seed(0)

window_size = 200
stride = 10

x = []
y_delta_l = []
y_delta_psi = []

imu_data_filenames = []
gt_data_filenames = []

imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu1.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu2.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu3.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu4.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu5.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu6.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu7.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/imu1.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/imu2.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/imu3.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/imu1.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/imu2.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/imu3.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/imu4.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/imu5.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu1.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu2.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu3.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu4.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu5.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu1.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu2.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu3.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu4.csv')

gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi1.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi2.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi3.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi4.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi5.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi6.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi7.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/vi1.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/vi2.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/vi3.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/vi1.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/vi2.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/vi3.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/vi4.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/vi5.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/vi1.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/vi2.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/vi3.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/vi4.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/vi5.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi1.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi2.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi3.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi4.csv')

for (cur_imu_data_filename, cur_gt_data_filename) in zip(imu_data_filenames, gt_data_filenames):
    cur_x, [cur_y_delta_l, cur_y_delta_psi], init_l, init_psi = load_dataset(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)
    x.append(cur_x)
    y_delta_l.append(cur_y_delta_l)
    y_delta_psi.append(cur_y_delta_psi)

x = np.vstack(x)
y_delta_l = np.vstack(y_delta_l)
y_delta_psi = np.vstack(y_delta_psi)

x, y_delta_l, y_delta_psi = shuffle(x, y_delta_l, y_delta_psi)

do_training = True

if do_training:
	model = create_model(window_size)

	#model_checkpoint = ModelCheckpoint('bidirectional_lstm.hdf5', monitor='loss', save_best_only=True, verbose=1)
	model_checkpoint = ModelCheckpoint('bidirectional_lstm.hdf5', monitor='val_loss', save_best_only=True, verbose=1)

	history = model.fit(x, [y_delta_l, y_delta_psi], epochs=500, batch_size=512, verbose=1, callbacks=[model_checkpoint], validation_split=0.1)

	#plt.plot(history.history['loss'])
	#plt.title('Model loss')
	#plt.ylabel('Loss')
	#plt.xlabel('Epoch')
	#plt.show()

	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.show()

model = load_model('bidirectional_lstm.hdf5')

x, [y_delta_l, y_delta_psi], init_l, init_psi = load_dataset('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/multi users/user2/syn/vi1.csv', window_size, stride)

[yhat_delta_l, yhat_delta_psi] = model.predict(x, batch_size=1, verbose=1)

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

gt_trajectory = generate_trajectory(init_l, init_psi, y_delta_l, y_delta_psi)
pred_trajectory = generate_trajectory(init_l, init_psi, yhat_delta_l, yhat_delta_psi)
pred_trajectory_only_l = generate_trajectory(init_l, init_psi, yhat_delta_l, y_delta_psi)
pred_trajectory_only_psi = generate_trajectory(init_l, init_psi, y_delta_l, yhat_delta_psi)

#gt_trajectory = generate_trajectory(init_l, init_psi, y_delta_l[::20], y_delta_psi[::20])
#pred_trajectory = generate_trajectory(init_l, init_psi, yhat_delta_l[::20], yhat_delta_psi[::20])
#pred_trajectory_only_l = generate_trajectory(init_l, init_psi, yhat_delta_l[::20], y_delta_psi[::20])
#pred_trajectory_only_psi = generate_trajectory(init_l, init_psi, y_delta_l[::20], yhat_delta_psi[::20])

plt.figure()
#plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1])
#plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1])
plt.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1])
plt.plot(pred_trajectory[0:200, 0], pred_trajectory[0:200, 1])
plt.title('Trajectory Pred vs Ground Truth')
plt.ylabel('Y (m)')
plt.xlabel('X (m)')
plt.legend(['Trajectory Ground Truth', 'Trajectory Pred'], loc='upper left')

plt.figure()
#plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1])
#plt.plot(pred_trajectory_only_l[:, 0], pred_trajectory_only_l[:, 1])
plt.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1])
plt.plot(pred_trajectory_only_l[0:200, 0], pred_trajectory_only_l[0:200, 1])
plt.title('Trajectory Pred Only Delta L vs Ground Truth')
plt.ylabel('Y (m)')
plt.xlabel('X (m)')
plt.legend(['Trajectory Ground Truth', 'Trajectory Pred Only Delta Psi'], loc='upper left')

plt.figure()
#plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1])
#plt.plot(pred_trajectory_only_psi[:, 0], pred_trajectory_only_psi[:, 1])
plt.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1])
plt.plot(pred_trajectory_only_psi[0:200, 0], pred_trajectory_only_psi[0:200, 1])
plt.title('Trajectory Pred Only Delta Psi vs Ground Truth')
plt.ylabel('Y (m)')
plt.xlabel('X (m)')
plt.legend(['Trajectory Ground Truth', 'Trajectory Pred Only Delta Psi'], loc='upper left')

plt.show()

np.savetxt('pred_positions.txt', pred_trajectory, delimiter=' ')