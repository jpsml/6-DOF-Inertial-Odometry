from dataset import *

x, y = load_dataset('Oxford Inertial Tracking Dataset/handheld/data1/syn/imu1.csv', 'Oxford Inertial Tracking Dataset/handheld/data1/syn/vi1.csv')

print('x[0]: ', x[0])
print('y[0]: ', y[0])

print('x[1]: ', x[1])
print('y[1]: ', y[1])

# TODO: recreate trajectory from loaded data