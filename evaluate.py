import numpy as np

from keras.models import load_model

from dataset import *
from util import *


#model_filename = 'bidirectional_lstm_mtl_pred_6D_handheld_all_seqs_1000_epochs.hdf5'
model_filename = 'bidirectional_lstm_6D_quat_handheld_all_seqs_400_epochs.hdf5'

model = load_model(model_filename)

window_size = 200
stride = 10

imu_data_filenames = []
gt_data_filenames = []

imu_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu1.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu2.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu3.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu4.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu5.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu6.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu7.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu8.csv')
imu_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/imu9.csv')

gt_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/vi1.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/vi2.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/vi3.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/vi4.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/vi5.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/vi6.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/vi7.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/vi8.csv')
gt_data_filenames.append('Oxford Inertial Tracking Dataset/multi users/user2/syn/vi9.csv')

print('evaluating trajectory rmse (m) using model ' + model_filename)

for (cur_imu_data_filename, cur_gt_data_filename) in zip(imu_data_filenames, gt_data_filenames):
	x, [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_imu_data_filename, cur_gt_data_filename, window_size, stride)
	
	[yhat_delta_p, yhat_delta_q] = model.predict(x, batch_size=1, verbose=0)

	gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
	pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

	#trajectory_rmse = np.sqrt(np.mean(np.square(pred_trajectory - gt_trajectory)))
	trajectory_rmse = np.sqrt(np.mean(np.square(pred_trajectory[0:200, :] - gt_trajectory[0:200, :])))

	print(trajectory_rmse)