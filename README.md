# Deep-Optimization-Prior
Deep Optimization Prior for THz Model Parameter Estimation

## How to download dataset

1. Download the source code
2. Extract the folder
3. Download dataset file from url https://www.cg.informatik.uni-siegen.de/data/Deep-Optimization-Prior/thz-dataset-modelparam.zip
4. Extract the folder "dataset" into the source code folder (or anywhere you want)

## How to train

There are two ways to train:

*Using Jupyter Notebook*
1. See [example.ipynb](example.ipynb)

*Using Python*
1. Prepare your dataset folder path (i.e., "./dataset")
2. Prepare your result folder path (i.e., "./dataset")
3. Set your desired optimizer (i.e., "unet")
4. Select your dataset (i.e., "metalpcb")
5. Select your learning rate (i.e., 0.01)
6. Pass all these parameters to `train.py`

For example:
`python train.py unet metalpcb --lr 0.01 --dataset_path ./dataset --dest_path ./result`

For more details, please check the shell script [example.sh](example.sh) or check the manual `python train.py --help`

## How to Cite
If you use this code in your scientific publication, please cite the paper

   **Deep Optimization Prior for THz Model Parameter Estimation**
     (T.M. Wong, H. Bauermeister, M. Kahl, P. Haring Bolivar, M. Moeller, A. Kolb),
     In Winter Conference on Applications of Computer Vision (WACV) 2022
