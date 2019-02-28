import numpy as np

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from dataset import *
from util import *
from model import *


#model_filename = 'bidirectional_lstm_pred.hdf5'
#model_filename = 'bidirectional_lstm.hdf5'
model_filename = 'bidirectional_lstm_mtl_pred_6D_quat_mult_loss_batch_size_32_500_epochs.hdf5'

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

print('evaluating trajectory rmse (m) using model ' + model_filename)

for (cur_imu_data_filename, cur_gt_data_filename) in zip(imu_data_filenames, gt_data_filenames):
    x, [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)
    #x, [y_delta_l, y_delta_theta, y_delta_psi], init_l, init_theta, init_psi = load_dataset_3d(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)

    #[yhat_delta_p, yhat_delta_q] = model.predict(x, batch_size=1, verbose=1)
    [yhat_delta_p, yhat_delta_q] = model.predict(x[0:200, :, :], batch_size=1, verbose=1)
    #[yhat_delta_l, yhat_delta_theta, yhat_delta_psi] = model.predict(x[0:200, :, :], batch_size=1, verbose=0)

    gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
    pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

    #gt_trajectory = generate_trajectory_3d(init_l, init_theta, init_psi, y_delta_l, y_delta_theta, y_delta_psi)
    #pred_trajectory = generate_trajectory_3d(init_l, init_theta, init_psi, yhat_delta_l, yhat_delta_theta, yhat_delta_psi)

	#trajectory_rmse = np.sqrt(np.mean(np.square(pred_trajectory -
	#gt_trajectory)))
    trajectory_rmse = np.sqrt(np.mean(np.square(pred_trajectory[0:200, :] - gt_trajectory[0:200, :])))

    print(trajectory_rmse)

    trajectory_length = 0

    for i in range(1, 200):
        trajectory_length += np.sqrt(np.sum(np.square(gt_trajectory[i, :] - gt_trajectory[i - 1, :]), axis=-1))

    print(trajectory_length)