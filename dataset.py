import numpy as np
from keras.utils import Sequence

def load_dataset(imu_data_filename, gt_data_filename, window_size=200, stride=10):

    imu_data = np.genfromtxt(imu_data_filename, delimiter=',')
    gt_data = np.genfromtxt(gt_data_filename, delimiter=',')
    
    gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    
    loc_data = gt_data[:, 2:4]

    l0 = loc_data[0, :]
    l1 = loc_data[window_size, :]

    l_diff = l1 - l0
    psi0 = np.arctan2(l_diff[1], l_diff[0])

    init_l = l1
    init_psi = psi0

    # TODO: store initial location and heading for every stride and use it for computing the trajectories
    #init_l = []
    #init_psi = []
    #prev_psi = []

    #for i in range(0, window_size, stride):        
    #    l0 = loc_data[i, :]
    #    l1 = loc_data[i + window_size, :]

    #    l_diff = l1 - l0
    #    psi0 = np.arctan2(l_diff[1], l_diff[0])

    #    init_l.append(np.array([l1])
    #    init_psi.append(np.array([psi0])
    #    prev_psi.append(np.array([psi0])

    x = []
    #y = []
    y_delta_l = []
    y_delta_psi = []

    for idx in range(stride, gyro_acc_data.shape[0] - window_size, stride):
        x.append(gyro_acc_data[idx : idx + window_size, :])

        l0 = loc_data[idx, :]
        l1 = loc_data[idx + window_size, :]

        l_diff = l1 - l0
        psi1 = np.arctan2(l_diff[1], l_diff[0])
        delta_l = np.linalg.norm(l_diff)
        delta_psi = psi1 - psi0

        psi0 = psi1

        if delta_psi < -np.pi:
            delta_psi += 2 * np.pi
        elif delta_psi > np.pi:
            delta_psi -= 2 * np.pi

        #y.append(np.array([delta_l, delta_psi]))

        y_delta_l.append(np.array([delta_l]))
        y_delta_psi.append(np.array([delta_psi]))

        #y_delta_l.append(np.array([delta_l / (window_size / 100)]))
        #y_delta_psi.append(np.array([delta_psi / (window_size / 100)]))


    x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    #y = np.reshape(y, (len(y), y[0].shape[0]))
    y_delta_l = np.reshape(y_delta_l, (len(y_delta_l), y_delta_l[0].shape[0]))
    y_delta_psi = np.reshape(y_delta_psi, (len(y_delta_psi), y_delta_psi[0].shape[0]))

    #return x, y, init_l, init_psi
    return x, [y_delta_l, y_delta_psi], init_l, init_psi
    #return x, y_delta_psi, init_l, init_psi