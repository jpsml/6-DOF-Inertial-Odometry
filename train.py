import numpy as np
import matplotlib.pyplot as plt
import cv2

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

from sklearn.utils import shuffle

from dataset import *
from model import *


def generate_trajectory_6d_rvec(init_rvec, init_tvec, y_delta_rvec, y_delta_tvec):
    cur_rvec = np.array(init_rvec)
    cur_tvec = np.array(init_tvec)    
    pred_tvec = []
    pred_tvec.append(np.array(cur_tvec))

    for [delta_rvec, delta_tvec] in zip(y_delta_rvec, y_delta_tvec):
        cv2.composeRT(cur_rvec, cur_tvec, delta_rvec, delta_tvec, cur_rvec, cur_tvec)
        pred_tvec.append(np.array(cur_tvec))

    return np.reshape(pred_tvec, (len(pred_tvec), 3))


def generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q):
    cur_p = np.array(init_p)
    cur_q = quaternion.from_float_array(init_q)
    pred_p = []
    pred_p.append(np.array(cur_p))

    for [delta_p, delta_q] in zip(y_delta_p, y_delta_q):
        cur_p = cur_p + np.matmul(quaternion.as_rotation_matrix(cur_q), delta_p.T).T
        cur_q = cur_q * quaternion.from_float_array(delta_q).normalized()
        pred_p.append(np.array(cur_p))

    return np.reshape(pred_p, (len(pred_p), 3))


def generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi):
    cur_l = np.array(init_l)
    cur_theta = np.array(init_theta)
    cur_psi = np.array(init_psi)
    pred_l = []
    pred_l.append(np.array(cur_l))

    for [delta_l, delta_theta, delta_psi] in zip(y_delta_l, y_delta_theta, y_delta_psi):
        cur_theta = cur_theta + delta_theta
        cur_psi = cur_psi + delta_psi
        cur_l[0] = cur_l[0] + delta_l * np.sin(cur_theta) * np.cos(cur_psi)
        cur_l[1] = cur_l[1] + delta_l * np.sin(cur_theta) * np.sin(cur_psi)
        cur_l[2] = cur_l[2] + delta_l * np.cos(cur_theta)
        pred_l.append(np.array(cur_l))

    return np.reshape(pred_l, (len(pred_l), 3))


def generate_trajectory_2d(init_l, init_psi, y_delta_l, y_delta_psi):
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

#y_delta_l = []
#y_delta_theta = []
#y_delta_psi = []

y_delta_p = []
y_delta_q = []

#y_delta_rvec = []
#y_delta_tvec = []

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

#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/imu1.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/imu2.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/imu3.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/imu4.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/imu5.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/imu6.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/imu7.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data2/syn/imu1.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data2/syn/imu2.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data2/syn/imu3.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data2/syn/imu4.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data2/syn/imu5.csv')

#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/vi1.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/vi2.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/vi3.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/vi4.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/vi5.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/vi6.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data1/syn/vi7.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data2/syn/vi1.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data2/syn/vi2.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data2/syn/vi3.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data2/syn/vi4.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/trolley/data2/syn/vi5.csv')

for i, (cur_imu_data_filename, cur_gt_data_filename) in enumerate(zip(imu_data_filenames, gt_data_filenames)):
    #cur_x, [cur_y_delta_l, cur_y_delta_psi], init_l, init_psi = load_dataset_2d(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)
    #cur_x, [cur_y_delta_l, cur_y_delta_theta, cur_y_delta_psi], init_l, init_theta, init_psi = load_dataset_3d(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)
    cur_x, [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)
    #cur_x, [cur_y_delta_rvec, cur_y_delta_tvec], init_rvec, init_tvec = load_dataset_6d_rvec(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)

    #plt.figure()
    #plt.plot(cur_y_delta_l)
    #plt.title('Delta L ' + str(i))
    #plt.ylabel('m')
    #plt.xlabel('time (0.1s)')

    #plt.figure()
    #plt.plot(cur_y_delta_theta)
    #plt.title('Delta Theta ' + str(i))
    #plt.ylabel('rad')
    #plt.xlabel('time (0.1s)')

    #plt.figure()
    #plt.plot(cur_y_delta_psi)
    #plt.title('Delta Psi ' + str(i))
    #plt.ylabel('rad')
    #plt.xlabel('time (0.1s)')

    #plt.show()

    x.append(cur_x)

    #y_delta_l.append(cur_y_delta_l)
    #y_delta_theta.append(cur_y_delta_theta)
    #y_delta_psi.append(cur_y_delta_psi)

    y_delta_p.append(cur_y_delta_p)
    y_delta_q.append(cur_y_delta_q)

    #y_delta_rvec.append(cur_y_delta_rvec)
    #y_delta_tvec.append(cur_y_delta_tvec)


x = np.vstack(x)

#y_delta_l = np.vstack(y_delta_l)
#y_delta_theta = np.vstack(y_delta_theta)
#y_delta_psi = np.vstack(y_delta_psi)

y_delta_p = np.vstack(y_delta_p)
y_delta_q = np.vstack(y_delta_q)

#_delta_rvec = np.vstack(y_delta_rvec)
#y_delta_tvec = np.vstack(y_delta_tvec)

#x, y_delta_l, y_delta_psi = shuffle(x, y_delta_l, y_delta_psi)
#x, y_delta_l, y_delta_theta, y_delta_psi = shuffle(x, y_delta_l, y_delta_theta, y_delta_psi)
x, y_delta_p, y_delta_q = shuffle(x, y_delta_p, y_delta_q)
#x, y_delta_rvec, y_delta_tvec = shuffle(x, y_delta_rvec, y_delta_tvec)

do_training = True

if do_training:
    #model = create_model_2d(window_size)
    #model = create_model_3d(window_size)
    model = create_model_6d_quat(window_size)
    #model = create_model_6d_rvec(window_size)

    #pred_model = create_pred_model_6d_quat(window_size)
    #train_model = create_train_model_6d_quat(pred_model, window_size)
    #train_model.compile(optimizer=Adam(0.0001), loss=None)

    model_checkpoint = ModelCheckpoint('bidirectional_lstm.hdf5', monitor='val_loss', save_best_only=True, verbose=1)

    #history = model.fit(x, [y_delta_l, y_delta_psi], epochs=400, batch_size=512, verbose=1, callbacks=[model_checkpoint], validation_split=0.1)
    #history = model.fit(x, [y_delta_l, y_delta_theta, y_delta_psi], epochs=400, batch_size=512, verbose=1, callbacks=[model_checkpoint], validation_split=0.1)
    history = model.fit(x, [y_delta_p, y_delta_q], epochs=400, batch_size=512, verbose=1, callbacks=[model_checkpoint], validation_split=0.1)
    #history = model.fit(x, [y_delta_rvec, y_delta_tvec], epochs=200, batch_size=512, verbose=1, callbacks=[model_checkpoint], validation_split=0.1)

    #history = train_model.fit([x, y_delta_p, y_delta_q], epochs=1000, batch_size=512, verbose=1, callbacks=[model_checkpoint], validation_split=0.1)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    #train_model = load_model('bidirectional_lstm.hdf5', custom_objects={'CustomMultiLossLayer':CustomMultiLossLayer}, compile=False)

    #print([K.get_value(log_var[0]) for log_var in train_model.layers[-1].log_vars])

model = load_model('bidirectional_lstm.hdf5')
#model = load_model('bidirectional_lstm.hdf5', custom_objects={'weighted_squared_error_xyz':weighted_squared_error_xyz, 'weighted_squared_error_wpqr':weighted_squared_error_wpqr})
#model = load_model('bidirectional_lstm_6D_quat_handheld_all_seqs_400_epochs.hdf5')
#model = load_model('bidirectional_lstm_6D_quat_mtl_optimal_weights_handheld_all_seqs_400_epochs.hdf5', custom_objects={'weighted_squared_error_xyz':weighted_squared_error_xyz, 'weighted_squared_error_wpqr':weighted_squared_error_wpqr})
#model = load_model('bidirectional_lstm_3D_handheld_all_seqs_400_epochs.hdf5')

#x, [y_delta_l, y_delta_psi], init_l, init_psi = load_dataset_2d('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/multi users/user2/syn/vi1.csv', window_size, stride)

#x, [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi = load_dataset_3d('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/multi users/user2/syn/vi1.csv', window_size, stride)

x, [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/multi users/user2/syn/vi1.csv', window_size, stride)
#x, [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu2.csv', 'Oxford Inertial Tracking Dataset/multi users/user2/syn/vi2.csv', window_size, stride)

#x, [y_delta_rvec, y_delta_tvec], init_rvec, init_tvec = load_dataset_6d_rvec('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/multi users/user2/syn/vi1.csv', window_size, stride)

#x, [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi = load_dataset_3d('Oxford Inertial Tracking Dataset/trolley/data2/syn/imu6.csv', 'Oxford Inertial Tracking Dataset/trolley/data2/syn/vi6.csv', window_size, stride)

#[yhat_delta_l, yhat_delta_psi] = model.predict(x, batch_size=1, verbose=1)
#[yhat_delta_l, yhat_delta_theta, yhat_delta_psi] = model.predict(x, batch_size=1, verbose=1)
[yhat_delta_p, yhat_delta_q] = model.predict(x, batch_size=1, verbose=1)
#[yhat_delta_rvec, yhat_delta_tvec] = model.predict(x, batch_size=1, verbose=1)

#plt.figure()
#plt.plot(y_delta_l)
#plt.plot(yhat_delta_l)
#plt.title('Delta L Pred vs Ground Truth')
#plt.ylabel('Delta L (m)')
#plt.xlabel('Time (0.1s)')
#plt.legend(['Delta L Ground Truth', 'Delta L Pred'], loc='upper left')

#plt.figure()
#plt.plot(y_delta_theta)
#plt.plot(yhat_delta_theta)
#plt.title('Delta Theta Pred vs Ground Truth')
#plt.ylabel('Delta Theta (rad)')
#plt.xlabel('Time (0.1s)')
#plt.legend(['Delta Theta Ground Truth', 'Delta Theta Pred'], loc='upper left')

#plt.figure()
#plt.plot(y_delta_psi)
#plt.plot(yhat_delta_psi)
#plt.title('Delta Psi Pred vs Ground Truth')
#plt.ylabel('Delta Psi (rad)')
#plt.xlabel('Time (0.1s)')
#plt.legend(['Delta Psi Ground Truth', 'Delta Psi Pred'], loc='upper left')

#plt.show()

#gt_trajectory = generate_trajectory_2d(init_l, init_psi, y_delta_l, y_delta_psi)
#pred_trajectory = generate_trajectory_2d(init_l, init_psi, yhat_delta_l, yhat_delta_psi)

#gt_trajectory = generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi)
#pred_trajectory = generate_trajectory_3d(init_l, init_theta, init_psi, yhat_delta_l, yhat_delta_theta, yhat_delta_psi)

gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

#gt_trajectory = generate_trajectory_6d_rvec(init_rvec, init_tvec, y_delta_rvec, y_delta_tvec)
#pred_trajectory = generate_trajectory_6d_rvec(init_rvec, init_tvec, yhat_delta_rvec, yhat_delta_tvec)

##plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1])
#plt.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1])
#plt.title('Trajectory Ground Truth')
#plt.ylabel('Y (m)')
#plt.xlabel('X (m)')
#plt.show()

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2])
##ax.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1], gt_trajectory[0:200, 2])
#ax.set_title('Trajectory Ground Truth');
#ax.set_xlabel('X (m)')
#ax.set_ylabel('Y (m)')
#ax.set_zlabel('Z (m)')
#plt.show()

#plt.figure()
#plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1])
#plt.plot(pred_trajectory[:, 0], pred_trajectory[:, 1])
#plt.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1])
#plt.plot(pred_trajectory[0:200, 0], pred_trajectory[0:200, 1])

#fig, ax = plt.subplots()
#plt.title('Trajectory Pred vs Ground Truth')
#plt.ylabel('Y (m)')
#plt.xlabel('X (m)')
#plt.legend(['Trajectory Ground Truth', 'Trajectory Pred'], loc='upper left')
#ax.set_xlim(np.minimum(np.amin(gt_trajectory[:, 0]), np.amin(pred_trajectory[:, 0])), np.maximum(np.amax(gt_trajectory[:, 0]), np.amax(pred_trajectory[:, 0])))
#ax.set_ylim(np.minimum(np.amin(gt_trajectory[:, 1]), np.amin(pred_trajectory[:, 1])), np.maximum(np.amax(gt_trajectory[:, 1]), np.amax(pred_trajectory[:, 1])))
#gt_x_data, gt_y_data, pred_x_data, pred_y_data = [], [], [], []
#gt_ln, = plt.plot([], [], animated=True)
#pred_ln, = plt.plot([], [], animated=True)
#def update_trajectories(frame):
#	gt_x_data.append(gt_trajectory[frame, 0])
#	gt_y_data.append(gt_trajectory[frame, 1])
#	pred_x_data.append(pred_trajectory[frame, 0])
#	pred_y_data.append(pred_trajectory[frame, 1])
#	gt_ln.set_data(gt_x_data, gt_y_data)
#	pred_ln.set_data(pred_x_data, pred_y_data)
#	return [gt_ln, pred_ln]
##ani = FuncAnimation(fig, update_trajectories, frames=gt_trajectory.shape[0], interval=100, blit=True)
#ani = FuncAnimation(fig, update_trajectories, frames=1200, interval=100, blit=True)

fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2])
#ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2])
ax.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1], gt_trajectory[0:200, 2])
ax.plot(pred_trajectory[0:200, 0], pred_trajectory[0:200, 1], pred_trajectory[0:200, 2])
ax.set_title('Trajectory Pred vs Ground Truth');
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
#min_lim = np.minimum(np.amin(gt_trajectory), np.amin(pred_trajectory))
#max_lim = np.maximum(np.amax(gt_trajectory), np.amax(pred_trajectory))
min_lim = np.minimum(np.amin(gt_trajectory[0:200, :]), np.amin(pred_trajectory[0:200, :]))
max_lim = np.maximum(np.amax(gt_trajectory[0:200, :]), np.amax(pred_trajectory[0:200, :]))
ax.set_xlim(min_lim, max_lim)
ax.set_ylim(min_lim, max_lim)
ax.set_zlim(min_lim, max_lim)
ax.legend(['Trajectory Ground Truth', 'Trajectory Pred'], loc='upper left')

plt.show()