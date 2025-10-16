# Super-Long Input Sequences for Long-Term Time Series Forecasting with Missing Values
![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg?style=plastic)
![PyTorch 2.1.0](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![CUDA 11.8](https://img.shields.io/badge/cuda-11.8-green.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the origin Pytorch implementation of SLNet in the following paper: 
[Super-Long Input Sequences for Long-Term Time Series Forecasting with Missing Values] (Accepted by IEEE Transactions on Instrumentation and Measurement). The data preprocessing, hyperparameter settings, experimental setups (including ablation studies), training duration, hardware specifications, and inference latency can be found in the manuscript.

## Model Architecture

<p align="center">
<img src="./img/SLNet.jpg" height = "480" width = "1550" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of the architecture of SLNet, which is composed of four main components: (a) an SLIS with adaptive multilevel segments for different
periodicities, (b) a focal encoder for efficient hierarchical feature extraction, (c) a shared dictionary for tokenizing and initializing predictions, and (d) a decoder
that fuses features and tokens to produce ultimate predictions and confidence scores.
</p>


## Requirements
- python == 3.11.4
- numpy == 1.24.4
- pandas == 1.5.3
- scipy == 1.10.1
- torch == 2.1.0+cu118
- scikit-learn == 1.4.2
- h5py == 3.7.0
- matplotlib == 3.7.1
- loguru == 0.7.2

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Raw Data
ECL was acquired at: [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). Solar dataset was acquired at: [Solar](https://drive.google.com/drive/folders/12ffxwxVAGM_MQiYpIk9aBLQrb2xQupT-). Wind was acquired at: [Wind]( https://www.kaggle.com/datasets/sohier/30-years-of-european-wind-generation). Hydro was acquired at: [Hydro](https://www.kaggle.com/datasets/mahbuburrahman2020/europe-green-electricity-generation-consumption).

### Data Preparation
We supply all processed datasets and put them under `./data`, the folder tree is shown below:
```
|-data
| |-ECL
| | |-ECL.csv
| |
| |-Hydro_BXX
| | |-Hydro_BXX.csv
| |
| |-Solar
| | |-solar_AL.csv
| |
| |-Wind
| | |-Wind.csv
| |
| ...
```

The processing details for the four datasets are as follows. We place ECL in the folder `./electricity` of [here](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing) (the folder tree in the link is shown as below) into folder `./data` and rename it from `./electricity` to `./ECL`. We rename the file of ECL from `electricity.csv` to `ECL.csv` and rename its last variable from `OT` to original `MT_321`. The processed file can be found at `./data/ECL/ECL.csv`
```
The folder tree in https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing:
|-autoformer
| |-electricity
| | |-electricity.csv
```
To standardize the data format, we convert the data file of [Solar](https://drive.google.com/drive/folders/12ffxwxVAGM_MQiYpIk9aBLQrb2xQupT-) from 'solar_AL.txt' to 'solar_AL.csv'. We place the processed file into the folder `./data/Solar`. For convenience, we processed the Wind and Hydro datasets and you can obtain the processed files at `./data/Wind/Wind.csv` and `./data/Hydro_BXX/Hydro_BXX.csv`, respectively.

## Usage
Commands for training and testing SLNet of all datasets are in `./Run.sh`. 

More parameter information please refer to `main.py`.

We provide a complete command for training and testing SLNet:

```
python -u main.py --data <data> --input_len <input_len> --pred_len <pred_len> --period <period> --encoder_layer <encoder_layer> --layer_stack <layer_stack> --MODWT_level<MODWT_level> --patch_size<patch_size> --ccc_number <ccc_number> --d_model <d_model> --learning_rate <learning_rate> --dropout <dropout> --batch_size <batch_size> --train_epochs <train_epochs> --itr <itr> --train --patience <patience> --decay<decay>
```

Here we provide a more detailed and complete command description for training and testing the model:

| Parameter name |                                          Description of parameter                                          |
|:--------------:|:----------------------------------------------------------------------------------------------------------:|
|      data      |                                              The dataset name                                              |
|   root_path    |                                       The root path of the data file                                       |
|   data_path    |                                             The data file name                                             |
|  checkpoints   |                                       Location of model checkpoints                                        |
|   input_len    |                                           Input sequence length                                            |
|    pred_len    |                                         Prediction sequence length                                         |
|    period    |                                         Vested periods for global timestamps                                         |
|     enc_in     |                                                 Input variable number                                                |
|    dec_out     |                                                Output variable number                                             |
|    d_model     |                                             Hidden dims of model                                             |
|  encoder_layer |                                            The number of stages                                            |
|   layer_stack  |                                       The number of layers per stage                                       |
|   patch_size   |                                The initial patch size in patch-wise attention                              |
|  MODWT_level   |                                           The level of MODWT/MRA                                           |
|  ccc_num  |                                           Number of correlated variables                                           |
|    dropout     |                                                  Dropout                                                   |
|    num_workers     |                                                  Data loader num workers                                                   |
|      itr       |                                             Experiments times                                              |
|  train_epochs  |                                      Train epochs of the second stage                                      |
|   batch_size   |                         The batch size of training input data                          |
|   decay   |                         Decay rate of learning rate per epoch                         |
|    patience    |                                          Early stopping patience                                           |
| learning_rate  |                                          Optimizer learning rate                                           |


## Results
The experiment parameters of each dataset are formated in the `./Run.sh`. You can refer to these parameters for experiments, and you can also adjust the parameters to obtain better mse results or draw better prediction figures. We present the multivariate forecasting results of the four datasets in Figure 2 (with missing values) and Figure 3 (without missing values).

<p align="center">
<img src="./img/result1.jpg" height = "300" alt="" align=center />
<br><br>
<b>Figure 2.</b> Forecasting results with missing values.
</p>

<p align="center">
<img src="./img/result2.jpg" height = "300" alt="" align=center />
<br><br>
<b>Figure 2.</b> Forecasting results without missing values..
</p>



## Contact
If you have any questions, feel free to contact Li Shen through Email (shenli@buaa.edu.cn) or Github issues. Pull requests are highly welcomed!
