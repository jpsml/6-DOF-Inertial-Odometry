import numpy as np
from keras.utils import Sequence

def load_dataset(imu_data_filename, gt_data_filename, window_size=200, stride=10):

    imu_data = np.genfromtxt(imu_data_filename, delimiter=',')
    gt_data = np.genfromtxt(gt_data_filename, delimiter=',')
    
    gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    
    loc_data = gt_data[:, 2:4]

    #l0 = loc_data[0, :]
    #l1 = loc_data[window_size, :]

    l0 = loc_data[window_size - stride - stride, :]
    l1 = loc_data[window_size - stride, :]
    
    l_diff = l1 - l0
    psi0 = np.arctan2(l_diff[1], l_diff[0])
    init_l = l1
    init_psi = psi0

    x = []
    y_delta_l = []
    y_delta_psi = []

    #for idx in range(stride, gyro_acc_data.shape[0] - window_size - 1, stride):
    #for idx in range(window_size, gyro_acc_data.shape[0] - window_size - 1, stride):
    for idx in range(0, gyro_acc_data.shape[0] - window_size - 1, stride):
        x.append(gyro_acc_data[idx + 1 : idx + 1 + window_size, :])

        #l0_diff = loc_data[idx, :] - loc_data[idx - stride, :]
        #l0_diff = loc_data[idx, :] - loc_data[idx - window_size, :]
        l0_diff = loc_data[idx + window_size - stride, :] - loc_data[idx + window_size - stride - stride, :]
        psi0 = np.arctan2(l0_diff[1], l0_diff[0])

        #l0 = loc_data[idx, :]
        l0 = loc_data[idx + window_size - stride, :]
        l1 = loc_data[idx + window_size, :]

        #l0 = loc_data[idx, :]
        #l1 = loc_data[idx + stride, :]

        l_diff = l1 - l0
        psi1 = np.arctan2(l_diff[1], l_diff[0])
        delta_l = np.linalg.norm(l_diff)
        delta_psi = psi1 - psi0

        #psi0 = psi1

        if delta_psi < -np.pi:
            delta_psi += 2 * np.pi
        elif delta_psi > np.pi:
            delta_psi -= 2 * np.pi

        y_delta_l.append(np.array([delta_l]))
        y_delta_psi.append(np.array([delta_psi]))

        #y_delta_l.append(np.array([delta_l / (window_size / 100)]))
        #y_delta_psi.append(np.array([delta_psi / (window_size / 100)]))


    x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    y_delta_l = np.reshape(y_delta_l, (len(y_delta_l), y_delta_l[0].shape[0]))
    y_delta_psi = np.reshape(y_delta_psi, (len(y_delta_psi), y_delta_psi[0].shape[0]))

    return x, [y_delta_l, y_delta_psi], init_l, init_psi