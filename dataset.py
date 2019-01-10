import numpy as np

def load_dataset(imu_data_filename, gt_data_filename, window_size = 200, stride = 10):

    imu_data = np.genfromtxt(imu_data_filename, delimiter=',')
    gt_data = np.genfromtxt(gt_data_filename, delimiter=',')
    
    gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    
    loc_data = gt_data[:, 2:4]

    x = []
    y = []

    l0 = loc_data[int(window_size/2) - int(stride/2) - stride, :]    
    l1 = loc_data[int(window_size/2) - int(stride/2), :]
    delta_l = l1 - l0
    psi0 = np.arctan2(delta_l[1], delta_l[0])

    for idx in range(0, gyro_acc_data.shape[0] - window_size, stride):
        x.append(gyro_acc_data[idx : idx + window_size])

        l0 = loc_data[idx + int(window_size/2) - int(stride/2), :]
        l1 = loc_data[idx + int(window_size/2) + int(stride/2), :]

        delta_l = l1 - l0
        psi1 = np.arctan2(delta_l[1], delta_l[0])
        delta_psi = psi1 - psi0

        if delta_psi < -np.pi:
            delta_psi += 2 * np.pi
        elif delta_psi > np.pi:
            delta_psi -= 2 * np.pi

        y.append(np.array([delta_l, delta_psi]))

        psi0 = psi1

    return x, y