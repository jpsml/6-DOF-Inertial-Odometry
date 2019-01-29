import numpy as np
import matplotlib.pyplot as plt
import cv2

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam

from sklearn.utils import shuffle

from time import time

from dataset import *
from model import *
from util import *


np.random.seed(0)

window_size = 200
stride = 10

x = []

y_delta_p = []
y_delta_q = []

imu_data_filenames = []
gt_data_filenames = []

imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu1.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu2.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu3.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu4.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu5.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu6.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu7.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/imu1.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/imu2.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/imu3.csv')
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
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu1.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu2.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu3.csv')
#imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu4.csv')

gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi1.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi2.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi3.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi4.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi5.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi6.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi7.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/vi1.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/vi2.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/vi3.csv')
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
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi1.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi2.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi3.csv')
#gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi4.csv')

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
    cur_x, [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)

    #plt.plot(cur_y_delta_p[:, 0])
    #plt.plot(cur_y_delta_p[:, 1])
    #plt.plot(cur_y_delta_p[:, 2])
    #plt.plot(cur_y_delta_q[:, 0])
    #plt.plot(cur_y_delta_q[:, 1])
    #plt.plot(cur_y_delta_q[:, 2])
    #plt.plot(cur_y_delta_q[:, 3])
    #plt.legend(['delta_p_x', 'delta_p_y', 'delta_p_z', 'delta_q_w', 'delta_q_x', 'delta_q_y', 'delta_q_z'], loc='upper left')
    #plt.show()

    x.append(cur_x)

    y_delta_p.append(cur_y_delta_p)
    y_delta_q.append(cur_y_delta_q)


x = np.vstack(x)

y_delta_p = np.vstack(y_delta_p)
y_delta_q = np.vstack(y_delta_q)

x, y_delta_p, y_delta_q = shuffle(x, y_delta_p, y_delta_q)

do_training = True

if do_training:
    model = create_model_6d_quat(window_size)

    #pred_model = create_pred_model_6d_quat(window_size)
    #train_model = create_train_model_6d_quat(pred_model, window_size)
    #train_model.compile(optimizer=Adam(0.0001), loss=None)

    model_checkpoint = ModelCheckpoint('bidirectional_lstm.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    #model_checkpoint = ModelCheckpoint('bidirectional_lstm_log_var.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    history = model.fit(x, [y_delta_p, y_delta_q], epochs=400, batch_size=512, verbose=1, callbacks=[model_checkpoint, tensorboard], validation_split=0.1)
    #history = train_model.fit([x, y_delta_p, y_delta_q], epochs=1000, batch_size=512, verbose=1, callbacks=[model_checkpoint, tensorboard], validation_split=0.1)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    #train_model = load_model('bidirectional_lstm_log_var.hdf5', custom_objects={'CustomMultiLossLayer':CustomMultiLossLayer}, compile=False)

    #print([K.get_value(log_var[0]) for log_var in train_model.layers[-1].log_vars])

    #pred_model = create_pred_model_6d_quat(window_size)
    #pred_model.set_weights(train_model.get_weights()[:-2])
    #pred_model.save('bidirectional_lstm_pred.hdf5')

#model = load_model('bidirectional_lstm.hdf5')
model = load_model('bidirectional_lstm.hdf5', custom_objects={'quaternion_multiplicative_error':quaternion_multiplicative_error})
#model = load_model('bidirectional_lstm_pred.hdf5')
#model = load_model('bidirectional_lstm_mtl_pred_6D_handheld_all_seqs_1000_epochs.hdf5')
#model = load_model('bidirectional_lstm_6D_quat_handheld_all_seqs_400_epochs.hdf5')

x, [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat('Oxford Inertial Tracking Dataset/handheld/data2/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/handheld/data2/syn/vi1.csv', window_size, stride)

[yhat_delta_p, yhat_delta_q] = model.predict(x, batch_size=1, verbose=1)

gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
##ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2])
#ax.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1], gt_trajectory[0:200, 2])
#ax.set_title('Trajectory Ground Truth');
#ax.set_xlabel('X (m)')
#ax.set_ylabel('Y (m)')
#ax.set_zlabel('Z (m)')
#plt.show()

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

#trajectory_rmse = np.sqrt(np.mean(np.square(pred_trajectory - gt_trajectory)))
trajectory_rmse = np.sqrt(np.mean(np.square(pred_trajectory[0:200, :] - gt_trajectory[0:200, :])))
print('trajectory rmse (m): ', trajectory_rmse)