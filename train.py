import numpy as np
import matplotlib.pyplot as plt
import argparse

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam

from sklearn.utils import shuffle

from time import time

from dataset import *
from model import *
from util import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['oxiod', 'euroc'], help='Training dataset name (\'oxiod\' or \'euroc\')')
    parser.add_argument('output', help='Model output name')
    args = parser.parse_args()

    np.random.seed(0)

    window_size = 200
    stride = 10

    x_gyro = []
    x_acc = []

    y_delta_p = []
    y_delta_q = []

    imu_data_filenames = []
    gt_data_filenames = []

    if args.dataset == 'oxiod':
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu3.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/imu1.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/imu2.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu2.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu4.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu4.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu2.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu7.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu4.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu5.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu3.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu2.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/imu3.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu1.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu3.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu5.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu4.csv')

        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi3.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/vi1.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/vi2.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi2.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi4.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi4.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi2.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi7.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi4.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi5.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi3.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi2.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data2/syn/vi3.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi1.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi3.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi5.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi4.csv')
    
    elif args.dataset == 'euroc':
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
        if args.dataset == 'oxiod':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_oxiod_dataset(cur_imu_data_filename, cur_gt_data_filename)
        elif args.dataset == 'euroc':
            cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data = load_euroc_mav_dataset(cur_imu_data_filename, cur_gt_data_filename)

        [cur_x_gyro, cur_x_acc], [cur_y_delta_p, cur_y_delta_q], init_p, init_q = load_dataset_6d_quat(cur_gyro_data, cur_acc_data, cur_pos_data, cur_ori_data, window_size, stride)

        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)

        y_delta_p.append(cur_y_delta_p)
        y_delta_q.append(cur_y_delta_q)

    x_gyro = np.vstack(x_gyro)
    x_acc = np.vstack(x_acc)

    y_delta_p = np.vstack(y_delta_p)
    y_delta_q = np.vstack(y_delta_q)

    x_gyro, x_acc, y_delta_p, y_delta_q = shuffle(x_gyro, x_acc, y_delta_p, y_delta_q)

    pred_model = create_pred_model_6d_quat(window_size)
    train_model = create_train_model_6d_quat(pred_model, window_size)
    train_model.compile(optimizer=Adam(0.0001), loss=None)

    model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    history = train_model.fit([x_gyro, x_acc, y_delta_p, y_delta_q], epochs=500, batch_size=32, verbose=1, callbacks=[model_checkpoint, tensorboard], validation_split=0.1)

    train_model = load_model('model_checkpoint.hdf5', custom_objects={'CustomMultiLossLayer':CustomMultiLossLayer}, compile=False)

    pred_model = create_pred_model_6d_quat(window_size)
    pred_model.set_weights(train_model.get_weights()[:-2])
    pred_model.save('%s.hdf5' % args.output)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()