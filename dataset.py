import numpy as np
from keras.utils import Sequence

def load_dataset(imu_data_filename, gt_data_filename, window_size=200, stride=10):

    imu_data = np.genfromtxt(imu_data_filename, delimiter=',')
    gt_data = np.genfromtxt(gt_data_filename, delimiter=',')
    
    gyro_acc_data = np.concatenate([imu_data[:, 4:7], imu_data[:, 10:13]], axis=1)
    
    loc_data = gt_data[:, 2:4]

    x = []
    y = []

    l0 = loc_data[window_size // 2 - stride // 2 - stride, :]    
    l1 = loc_data[window_size // 2 - stride // 2, :]
    l_diff = l1 - l0
    psi0 = np.arctan2(l_diff[1], l_diff[0])

    init_l = l1
    init_psi = psi0

    for idx in range(0, gyro_acc_data.shape[0] - window_size, stride):
        x.append(gyro_acc_data[idx : idx + window_size, :])

        l0 = loc_data[idx + window_size // 2 - stride // 2, :]
        l1 = loc_data[idx + window_size // 2 + stride // 2, :]

        l_diff = l1 - l0
        psi1 = np.arctan2(l_diff[1], l_diff[0])
        delta_l = np.linalg.norm(l_diff)
        delta_psi = psi1 - psi0

        if delta_psi < -np.pi:
            delta_psi += 2 * np.pi
        elif delta_psi > np.pi:
            delta_psi -= 2 * np.pi

        y.append(np.array([delta_l, delta_psi]))

        psi0 = psi1

    x = np.reshape(x, (len(x), x[0].shape[0], x[0].shape[1]))
    y = np.reshape(y, (len(y), y[0].shape[0]))

    return x, y, init_l, init_psi


def train_data_generator(x, y, batch_size=32):
    while True:
        num_full_batches = x.shape[0] // batch_size

        for i in range(0, num_full_batches):
            yield (x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])

        if (x.shape[0] % batch_size != 0):
            yield (x[num_full_batches * batch_size:], y[num_full_batches * batch_size:])

def test_data_generator(x, batch_size=32):

    num_full_batches = x.shape[0] // batch_size

    for i in range(0, num_full_batches):
        yield x[i * batch_size:(i + 1) * batch_size]

    if (x.shape[0] % batch_size != 0):
        yield x[num_full_batches * batch_size:]