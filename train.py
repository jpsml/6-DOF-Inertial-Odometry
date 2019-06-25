import numpy as np
import matplotlib.pyplot as plt

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

do_training = False

if do_training:
    #x = []
    x_gyro = []
    x_acc = []

    y_delta_p = []
    y_delta_q = []

    #y_delta_l = []
    #y_delta_theta = []
    #y_delta_psi = []

    imu_data_filenames = []
    gt_data_filenames = []

    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu3.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/imu1.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/imu2.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu2.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/imu4.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu4.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu2.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu7.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu4.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu5.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu3.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/imu2.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/imu3.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu1.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/imu3.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/imu5.csv')
    #imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu4.csv')

    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi3.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/vi1.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/vi2.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi2.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/vi4.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/vi4.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/vi2.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi7.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi4.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/vi5.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi3.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/vi2.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data2/syn/vi3.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi1.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/vi3.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/vi5.csv')
    #gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi4.csv')

    imu_data_filenames.append('MH_01_easy/mav0/imu0/data.csv')
    imu_data_filenames.append('MH_03_medium/mav0/imu0/data.csv')
    imu_data_filenames.append('MH_05_difficult/mav0/imu0/data.csv')
    imu_data_filenames.append('V1_02_medium/mav0/imu0/data.csv')
    imu_data_filenames.append('V2_01_easy/mav0/imu0/data.csv')
    imu_data_filenames.append('V2_03_difficult/mav0/imu0/data.csv')

    gt_data_filenames.append('MH_01_easy/mav0/state_groundtruth_estimate0/data.csv')
    gt_data_filenames.append('MH_03_medium/mav0/state_groundtruth_estimate0/data.csv')
    gt_data_filenames.append('MH_05_difficult/mav0/state_groundtruth_estimate0/data.csv')
    gt_data_filenames.append('V1_02_medium/mav0/state_groundtruth_estimate0/data.csv')
    gt_data_filenames.append('V2_01_easy/mav0/state_groundtruth_estimate0/data.csv')
    gt_data_filenames.append('V2_03_difficult/mav0/state_groundtruth_estimate0/data.csv')

    for i, (cur_imu_data_filename, cur_gt_data_filename) in enumerate(zip(imu_data_filenames, gt_data_filenames)):        
        #cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)
        cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_euroc_mav_dataset(cur_imu_data_filename, cur_gt_data_filename)
        #cur_x, [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data, window_size, stride)
        [cur_x_gyro, cur_x_acc], [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data, window_size, stride)
        #cur_x, [cur_y_delta_l, cur_y_delta_theta, cur_y_delta_psi], init_l, init_theta, init_psi = load_dataset_3d(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)

        #plt.plot(cur_y_delta_p[:, 0])
        #plt.plot(cur_y_delta_p[:, 1])
        #plt.plot(cur_y_delta_p[:, 2])
        #plt.plot(cur_y_delta_q[:, 0])
        #plt.plot(cur_y_delta_q[:, 1])
        #plt.plot(cur_y_delta_q[:, 2])
        #plt.plot(cur_y_delta_q[:, 3])
        #plt.legend(['delta_p_x', 'delta_p_y', 'delta_p_z', 'delta_q_w', 'delta_q_x', 'delta_q_y', 'delta_q_z'], loc='upper left')
        #plt.show()

        #x.append(cur_x)
        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)

        y_delta_p.append(cur_y_delta_p)
        y_delta_q.append(cur_y_delta_q)

        #y_delta_l.append(cur_y_delta_l)
        #y_delta_theta.append(cur_y_delta_theta)
        #y_delta_psi.append(cur_y_delta_psi)


    #x = np.vstack(x)
    x_gyro = np.vstack(x_gyro)
    x_acc = np.vstack(x_acc)

    y_delta_p = np.vstack(y_delta_p)
    y_delta_q = np.vstack(y_delta_q)

    #y_delta_l = np.vstack(y_delta_l)
    #y_delta_theta = np.vstack(y_delta_theta)
    #y_delta_psi = np.vstack(y_delta_psi)

    #x, y_delta_p, y_delta_q = shuffle(x, y_delta_p, y_delta_q)
    x_gyro, x_acc, y_delta_p, y_delta_q = shuffle(x_gyro, x_acc, y_delta_p, y_delta_q)
    #x, y_delta_l, y_delta_theta, y_delta_psi = shuffle(x, y_delta_l, y_delta_theta, y_delta_psi)

    #model = create_model_6d_quat(window_size)

    pred_model = create_pred_model_6d_quat(window_size)
    train_model = create_train_model_6d_quat(pred_model, window_size)
    #pred_model = create_pred_model_3d(window_size)
    #train_model = create_train_model_3d(pred_model, window_size)
    train_model.compile(optimizer=Adam(0.0001), loss=None)

    #model_checkpoint = ModelCheckpoint('bidirectional_lstm.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    model_checkpoint = ModelCheckpoint('bidirectional_lstm_log_var.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    #history = model.fit(x, [y_delta_p, y_delta_q], epochs=400, batch_size=512, verbose=1, callbacks=[model_checkpoint, tensorboard], validation_split=0.1)
    #history = train_model.fit([x, y_delta_p, y_delta_q], epochs=500, batch_size=32, verbose=1, callbacks=[model_checkpoint, tensorboard], validation_split=0.1)
    history = train_model.fit([x_gyro, x_acc, y_delta_p, y_delta_q], epochs=500, batch_size=32, verbose=1, callbacks=[model_checkpoint, tensorboard], validation_split=0.1)
    #history = train_model.fit([x, y_delta_l, y_delta_theta, y_delta_psi], epochs=500, batch_size=32, verbose=1, callbacks=[model_checkpoint, tensorboard], validation_split=0.1)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    train_model = load_model('bidirectional_lstm_log_var.hdf5', custom_objects={'CustomMultiLossLayer':CustomMultiLossLayer}, compile=False)
    print([K.get_value(log_var[0]) for log_var in train_model.layers[-1].log_vars])

    pred_model = create_pred_model_6d_quat(window_size)
    pred_model.set_weights(train_model.get_weights()[:-2])
    #pred_model = create_pred_model_3d(window_size)
    #pred_model.set_weights(train_model.get_weights()[:-3])
    pred_model.save('bidirectional_lstm_pred.hdf5')

model = load_model('bidirectional_lstm_pred.hdf5')
#model = load_model('bidirectional_lstm.hdf5', custom_objects={'quaternion_mean_multiplicative_error':quaternion_mean_multiplicative_error})
#model = load_model('bidirectional_lstm.hdf5', custom_objects={'quaternion_log_phi_4_error':quaternion_phi_4_error})
#model = load_model('bidirectional_lstm_mtl_pred_6D_quat_mult_loss_batch_size_32_500_epochs.hdf5')

#gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/handheld/data4/syn/vi1.csv')
gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset('MH_04_difficult/mav0/imu0/data.csv', 'MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv')
#x, [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
[x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
#x, [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi = load_dataset_3d('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu3.csv', 'Oxford Inertial Tracking Dataset/handheld/data4/syn/vi3.csv', window_size, stride)

#[yhat_delta_p, yhat_delta_q] = model.predict(x, batch_size=1, verbose=1)
#[yhat_delta_p, yhat_delta_q] = model.predict(x[0:200, :, :], batch_size=1, verbose=1)
[yhat_delta_p, yhat_delta_q] = model.predict([x_gyro, x_acc], batch_size=1, verbose=1)
#[yhat_delta_p, yhat_delta_q] = model.predict([x_gyro[0:200, :, :], x_acc[0:200, :, :]], batch_size=1, verbose=1)
#[yhat_delta_l, yhat_delta_theta, yhat_delta_psi] = model.predict(x, batch_size=1, verbose=1)

gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

#gt_trajectory = generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi)
#pred_trajectory = generate_trajectory_3d(init_l, init_theta, init_psi, yhat_delta_l, yhat_delta_theta, yhat_delta_psi)

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
#ani = FuncAnimation(fig, update_trajectories, frames=pred_trajectory.shape[0], interval=100, blit=True)
#plt.show()

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2])
#ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2])
##ax.plot(gt_trajectory[0:200, 0], gt_trajectory[0:200, 1], gt_trajectory[0:200, 2])
##ax.plot(pred_trajectory[0:200, 0], pred_trajectory[0:200, 1], pred_trajectory[0:200, 2])
##ax.set_title('Trajectory Pred vs Ground Truth');
#ax.set_xlabel('X (m)')
#ax.set_ylabel('Y (m)')
#ax.set_zlabel('Z (m)')
#min_x = np.minimum(np.amin(gt_trajectory[:, 0]), np.amin(pred_trajectory[:, 0]))
#min_y = np.minimum(np.amin(gt_trajectory[:, 1]), np.amin(pred_trajectory[:, 1]))
#min_z = np.minimum(np.amin(gt_trajectory[:, 2]), np.amin(pred_trajectory[:, 2]))
#max_x = np.maximum(np.amax(gt_trajectory[:, 0]), np.amax(pred_trajectory[:, 0]))
#max_y = np.maximum(np.amax(gt_trajectory[:, 1]), np.amax(pred_trajectory[:, 1]))
#max_z = np.maximum(np.amax(gt_trajectory[:, 2]), np.amax(pred_trajectory[:, 2]))
##min_x = np.minimum(np.amin(gt_trajectory[0:200, 0]), np.amin(pred_trajectory[0:200, 0]))
##min_y = np.minimum(np.amin(gt_trajectory[0:200, 1]), np.amin(pred_trajectory[0:200, 1]))
##min_z = np.minimum(np.amin(gt_trajectory[0:200, 2]), np.amin(pred_trajectory[0:200, 2]))
##max_x = np.maximum(np.amax(gt_trajectory[0:200, 0]), np.amax(pred_trajectory[0:200, 0]))
##max_y = np.maximum(np.amax(gt_trajectory[0:200, 1]), np.amax(pred_trajectory[0:200, 1]))
##max_z = np.maximum(np.amax(gt_trajectory[0:200, 2]), np.amax(pred_trajectory[0:200, 2]))
#range_x = np.absolute(max_x - min_x)
#range_y = np.absolute(max_y - min_y)
#range_z = np.absolute(max_z - min_z)
#max_range = np.maximum(np.maximum(range_x, range_y), range_z)
#ax.set_xlim(min_x, min_x + max_range)
#ax.set_ylim(min_y, min_y + max_range)
#ax.set_zlim(min_z, min_z + max_range)
#ax.legend(['ground truth', 'predicted'], loc='upper right')
#plt.show()

min_x = np.minimum(np.amin(gt_trajectory[:, 0]), np.amin(pred_trajectory[:, 0]))
min_y = np.minimum(np.amin(gt_trajectory[:, 1]), np.amin(pred_trajectory[:, 1]))
min_z = np.minimum(np.amin(gt_trajectory[:, 2]), np.amin(pred_trajectory[:, 2]))
max_x = np.maximum(np.amax(gt_trajectory[:, 0]), np.amax(pred_trajectory[:, 0]))
max_y = np.maximum(np.amax(gt_trajectory[:, 1]), np.amax(pred_trajectory[:, 1]))
max_z = np.maximum(np.amax(gt_trajectory[:, 2]), np.amax(pred_trajectory[:, 2]))
#min_x = np.minimum(np.amin(gt_trajectory[0:200, 0]), np.amin(pred_trajectory[0:200, 0]))
#min_y = np.minimum(np.amin(gt_trajectory[0:200, 1]), np.amin(pred_trajectory[0:200, 1]))
#min_z = np.minimum(np.amin(gt_trajectory[0:200, 2]), np.amin(pred_trajectory[0:200, 2]))
#max_x = np.maximum(np.amax(gt_trajectory[0:200, 0]), np.amax(pred_trajectory[0:200, 0]))
#max_y = np.maximum(np.amax(gt_trajectory[0:200, 1]), np.amax(pred_trajectory[0:200, 1]))
#max_z = np.maximum(np.amax(gt_trajectory[0:200, 2]), np.amax(pred_trajectory[0:200, 2]))
range_x = np.absolute(max_x - min_x)
range_y = np.absolute(max_y - min_y)
range_z = np.absolute(max_z - min_z)
max_range = np.maximum(np.maximum(range_x, range_y), range_z)
for num in range(1, pred_trajectory.shape[0]):
    fig = plt.figure(figsize=[19.2, 10.8], dpi=100)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(min_x, min_x + max_range)
    ax.set_ylim(min_y, min_y + max_range)
    ax.set_zlim(min_z, min_z + max_range)
    gt_ln1, = ax.plot(gt_trajectory[:num, 0], gt_trajectory[:num, 1], gt_trajectory[:num, 2])
    red_ln1, = ax.plot(pred_trajectory[:num, 0], pred_trajectory[:num, 1], pred_trajectory[:num, 2])
    ax.view_init(elev=75, azim=0)
    ax.legend(['ground truth', 'predicted'], loc='upper right')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim(min_x, min_x + max_range)
    ax.set_ylim(min_y, min_y + max_range)
    ax.set_zlim(min_z, min_z + max_range)
    gt_ln2, = ax.plot(gt_trajectory[:num, 0], gt_trajectory[:num, 1], gt_trajectory[:num, 2])
    pred_ln2, = ax.plot(pred_trajectory[:num, 0], pred_trajectory[:num, 1], pred_trajectory[:num, 2])
    ax.view_init(elev=15, azim=0)
    ax.legend(['ground truth', 'predicted'], loc='upper right')
    def update_trajectories(num):
        gt_ln1.set_data(gt_trajectory[:num, 0], gt_trajectory[:num, 1])
        gt_ln1.set_3d_properties(gt_trajectory[:num, 2])
        pred_ln1.set_data(pred_trajectory[:num, 0], pred_trajectory[:num, 1])
        pred_ln1.set_3d_properties(pred_trajectory[:num, 2])
        gt_ln2.set_data(gt_trajectory[:num, 0], gt_trajectory[:num, 1])
        gt_ln2.set_3d_properties(gt_trajectory[:num, 2])
        pred_ln2.set_data(pred_trajectory[:num, 0], pred_trajectory[:num, 1])
        pred_ln2.set_3d_properties(pred_trajectory[:num, 2])
        return [gt_ln1, pred_ln1, gt_ln2, pred_ln2]
    #ani = FuncAnimation(fig, update_trajectories, frames=pred_trajectory.shape[0], interval=100, blit=True)
    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())
    plt.draw()
    plt.savefig('frame_{:04d}.png'.format(num))
    plt.close()
    #plt.show()

trajectory_rmse = np.sqrt(np.mean(np.square(pred_trajectory - gt_trajectory)))
#trajectory_rmse = np.sqrt(np.mean(np.square(pred_trajectory[0:200, :] - gt_trajectory[0:200, :])))
print('trajectory rmse (m): ', trajectory_rmse)

#x_rmse = np.sqrt(np.mean(np.square(pred_trajectory[0:200, 0] - gt_trajectory[0:200, 0])))
#print('x rmse (m): ', x_rmse)

#y_rmse = np.sqrt(np.mean(np.square(pred_trajectory[0:200, 1] - gt_trajectory[0:200, 1])))
#print('y rmse (m): ', y_rmse)

#z_rmse = np.sqrt(np.mean(np.square(pred_trajectory[0:200, 2] - gt_trajectory[0:200, 2])))
#print('z rmse (m): ', z_rmse)