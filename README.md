# IMU-Based 6-DOF Odometry

By [João Paulo Lima](https://www.cin.ufpe.br/~jpsml/), [Hideaki Uchiyama](http://limu.ait.kyushu-u.ac.jp/~uchiyama/me/index_e.html), [Rin-ichiro Taniguchi](http://limu.ait.kyushu-u.ac.jp/e/member/member0001.html).

This repository contains the code for the paper "[End-to-End Learning Framework for IMU-Based 6-DOF Odometry](https://www.mdpi.com/1424-8220/19/17/3777)". You can find a demonstration video [here](https://www.mdpi.com/1424-8220/19/17/3777/s2).

## Prerequisites
- Python 3
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn
- Pandas
- SciPy
- numpy-quaternion
- tfquaternion

## Training

We provide training code that can use [OxIOD](http://deepio.cs.ox.ac.uk/) or [EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) datasets.

1. Download the desired dataset and unzip it into the project folder (the path should be `"<project folder>/Oxford Inertial Odometry Dataset/handheld/data<id>/"` for OxIOD and `"<project folder>/<sequence name>/mav0/"` for EuRoC MAV)
2. Run `python train.py dataset output`, where `dataset` is either `oxiod` or `euroc` and `output` is the model output name (`output.hdf5`).

## Pretrained models

Pretrained models can be downloaded here:
- [OxIOD](https://drive.google.com/uc?export=download&id=1xBsOJXVUc_cO1ybzJJxBq5L36M49TMsE)
- [EuRoC MAV](https://drive.google.com/uc?export=download&id=1dXJasJx3jpS6LiqrvP2BpHSCgqLn85nX)

## Testing

We provide code for trajectory prediction and visual comparison with ground truth trajectories from [OxIOD](http://deepio.cs.ox.ac.uk/) or [EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) datasets.

1. Download the desired dataset and unzip it into the project folder (the path should be `"<project folder>/Oxford Inertial Odometry Dataset/handheld/data<id>/"` for OxIOD and `"<project folder>/<sequence name>/mav0/"` for EuRoC MAV)
2. Run `python test.py dataset model input gt`, where:
- `dataset` is either `oxiod` or `euroc`;
- `model` is the trained model file path (e.g. `6dofio_oxiod.hdf5`);
- `input` is the input sequence path (e.g. `"Oxford Inertial Odometry Dataset/handheld/data4/syn/imu1.csv"` for OxIOD, `"MH_02_easy/mav0/imu0/data.csv\"` for EuRoC MAV);
- `gt` is the ground truth path (e.g. `"Oxford Inertial Odometry Dataset/handheld/data4/syn/vi1.csv"` for OxIOD, `"MH_02_easy/mav0/state_groundtruth_estimate0/data.csv"` for EuRoC MAV).

## Evaluation

We provide code for computing trajectory RMSE for testing sequences from [OxIOD](http://deepio.cs.ox.ac.uk/) or [EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) datasets.

1. Download the desired dataset and unzip it into the project folder (the path should be `"<project folder>/Oxford Inertial Odometry Dataset/handheld/data<id>/"` for OxIOD and `"<project folder>/<sequence name>/mav0/"` for EuRoC MAV)
2. Run `python evaluate.py dataset model`, where `dataset` is either `oxiod` or `euroc` and `model` is the trained model file path (e.g. `6dofio_oxiod.hdf5`).

## Citation

If you use this method in your research, please cite:

    @article{lima2019end,
            title={End-to-End Learning Framework for IMU-Based 6-DOF Odometry},
            author={Silva do Monte Lima, Jo{\~a}o Paulo and Uchiyama, Hideaki and Taniguchi, Rin-ichiro},
            journal={Sensors},
            volume={19},
            number={17},
            pages={3777},
            year={2019},
            publisher={Multidisciplinary Digital Publishing Institute}
    }

## License

BSD
