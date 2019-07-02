import numpy as np

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from dataset import *
from util import *
from model import *


model_filename = 'bidirectional_lstm_pred.hdf5'
#model_filename = 'bidirectional_lstm_mtl_pred_6D_quat_mult_loss_conv_layers_batch_size_32_500_epochs_window_size_200_stride_10.hdf5'
#model_filename = 'bidirectional_lstm_pred_euroc_kernel_size_17.hdf5'

model = load_model(model_filename)
#model = load_model('bidirectional_lstm.hdf5', custom_objects={'quaternion_mean_multiplicative_error':quaternion_mean_multiplicative_error})
#model = load_model('bidirectional_lstm.hdf5', custom_objects={'quaternion_log_phi_4_error':quaternion_log_phi_4_error})

window_size = 200
stride = 10

imu_data_filenames = []
gt_data_filenames = []

imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu2.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu5.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu6.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/imu1.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu1.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/imu3.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/imu1.csv')

gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi2.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi5.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data1/syn/vi6.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data3/syn/vi1.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/vi1.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data4/syn/vi3.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/handheld/data5/syn/vi1.csv')

#imu_data_filenames.append('MH_02_easy/mav0/imu0/data.csv')
#imu_data_filenames.append('MH_04_difficult/mav0/imu0/data.csv')
#imu_data_filenames.append('V1_03_difficult/mav0/imu0/data.csv')
#imu_data_filenames.append('V2_02_medium/mav0/imu0/data.csv')
#imu_data_filenames.append('V1_01_easy/mav0/imu0/data.csv')

#gt_data_filenames.append('MH_02_easy/mav0/state_groundtruth_estimate0/data.csv')
#gt_data_filenames.append('MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv')
#gt_data_filenames.append('V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv')
#gt_data_filenames.append('V2_02_medium/mav0/state_groundtruth_estimate0/data.csv')
#gt_data_filenames.append('V1_01_easy/mav0/state_groundtruth_estimate0/data.csv')

print('evaluating trajectory rmse (m) using model ' + model_filename)

all_y_delta_p = []
all_yhat_delta_p = []

for (cur_imu_data_filename, cur_gt_data_filename) in zip(imu_data_filenames, gt_data_filenames):
    gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)
    #gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset(cur_imu_data_filename, cur_gt_data_filename)
    #x, [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
    [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
    #x, [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi = load_dataset_3d(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)

    #[yhat_delta_p, yhat_delta_q] = model.predict(x, batch_size=1, verbose=1)
    #[yhat_delta_p, yhat_delta_q] = model.predict(x[0:200, :, :], batch_size=1, verbose=0)
    #[yhat_delta_p, yhat_delta_q] = model.predict([x_gyro, x_acc], batch_size=1, verbose=0)
    [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro[0:200, :, :], x_acc[0:200, :, :]], batch_size=1, verbose=0)
    #[yhat_delta_l, yhat_delta_theta, yhat_delta_psi] = model.predict(x[0:200, :, :], batch_size=1, verbose=0)

    all_y_delta_p.append(y_delta_p)
    all_yhat_delta_p.append(yhat_delta_p)

    gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
    pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

    #gt_trajectory = generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi)
    #pred_trajectory = generate_trajectory_3d(init_l, init_theta, init_psi, yhat_delta_l, yhat_delta_theta, yhat_delta_psi)

    #trajectory_rmse = np.sqrt(np.mean(np.square(pred_trajectory - gt_trajectory)))
    trajectory_rmse = np.sqrt(np.mean(np.square(pred_trajectory[0:200, :] - gt_trajectory[0:200, :])))

    print(trajectory_rmse)

    #trajectory_length = 0
    #for i in range(1, 200):
    #    trajectory_length += np.sqrt(np.sum(np.square(gt_trajectory[i, :] - gt_trajectory[i - 1, :]), axis=-1))
    #print(trajectory_length)

    #y_delta_p_norm = np.linalg.norm(y_delta_p, axis=-1)
    #yhat_delta_p_norm = np.linalg.norm(yhat_delta_p, axis=-1)

    #trans_mae = np.mean(np.absolute(y_delta_p_norm - yhat_delta_p_norm))
    #trans_rmse = np.sqrt(np.mean(np.square(y_delta_p_norm - yhat_delta_p_norm)))

    #print('trans_mae: ', trans_mae)
    #print('trans_rmse: ', trans_rmse)

#all_y_delta_p = np.vstack(all_y_delta_p)
#all_yhat_delta_p = np.vstack(all_yhat_delta_p)

#y_delta_p_norm = np.linalg.norm(all_y_delta_p, axis=-1)
#yhat_delta_p_norm = np.linalg.norm(all_yhat_delta_p, axis=-1)

#trans_mae = np.mean(np.absolute(y_delta_p_norm - yhat_delta_p_norm))
#trans_rmse = np.sqrt(np.mean(np.square(y_delta_p_norm - yhat_delta_p_norm)))

#print('trans_mae: ', trans_mae)
#print('trans_rmse: ', trans_rmse)