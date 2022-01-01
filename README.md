# Deep-Optimization-Prior
_**Deep Image Prior for THz Model Parameter Estimation**_

_Abstract_
<p align="justify">In this paper, we propose a deep optimization prior approach with application to the estimation of material-related model parameters from terahertz (THz) data that is acquired using a Frequency Modulated Continuous Wave (FMCW) THz scanning system. A stable estimation of the THz model parameters for low SNR and shot noise configurations is essential to achieve acquisition times required for applications in, e.g., quality control.</p>

<p align="justify">Conceptually, our deep optimization prior approach estimates the desired THz model parameters by optimizing for the weights of a neural network. While such a technique was shown to improve the reconstruction quality for convex objectives in the seminal work of Ulyanov et. al., our paper demonstrates that deep priors also allow to find better local optima in the non-convex energy landscape of the nonlinear inverse problem arising from THz imaging.</p>

<p align="justify">We verify this claim numerically on various THz parameter estimation problems for synthetic and real data under low SNR and shot noise conditions.  While the low SNR scenario not even requires regularization, the impact of shot noise is significantly reduced by total variation (TV) regularization. We compare our approach with existing optimization techniques that require sophisticated physically motivated initialization, and with a 1D single-pixel reparametrization method.</p>

## How to download paper

The paper is available at:<br/>
https://openaccess.thecvf.com/content/WACV2022/html/Wong_Deep_Optimization_Prior_for_THz_Model_Parameter_Estimation_WACV_2022_paper.html

## How to watch presentation

The presentation video is available on Youtube:<br/>
https://www.youtube.com/watch?v=6JP-q5C9E_4

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
1. Prepare your dataset folder path (e.g. `./dataset`)
2. Prepare your result folder path (e.g. `./result`)
3. Set your desired optimizer (e.g. `unet`)
4. Select your dataset (e.g. `metalpcb`)
5. Select your learning rate (e.g. `0.01`)
6. Pass all these parameters to `train.py`

For example:
`python train.py unet metalpcb --lr 0.01 --dataset_path ./dataset --dest_path ./result`

For more details, please check the shell script [example.sh](example.sh) or check the manual `python train.py --help`

## How to Cite
If you use this code in your scientific publication, please cite the paper

   **Deep Optimization Prior for THz Model Parameter Estimation**<br/>
     *Tak Ming Wong, Hartmut Bauermeister, Matthias Kahl, Peter Haring Bolívar, Michael Möller, Andreas Kolb,*<br/>
     Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 3811-3820
     
     @InProceedings{Wong_2022_WACV,
        author    = {Wong, Tak Ming and Bauermeister, Hartmut and Kahl, Matthias and Bol{\'\i}var, Peter Haring and M\"oller, Michael and Kolb, Andreas},
        title     = {Deep Optimization Prior for THz Model Parameter Estimation},
        booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
        month     = {January},
        year      = {2022},
        pages     = {3811-3820}
    }
