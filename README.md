# Binaural-audio-generation
## Environment
* Python=3.6 
* Pytorch=1.6.0

## Dataset


**FAIR-Play** can be accessed [here](https://github.com/facebookresearch/FAIR-Play).
**YT-ASMR** can be accessed [here](https://github.com/karreny/telling-left-from-right/tree/dataset).


## Training and Ealuation

1. Prepare datasets. Please prepare the dataset as the instructions in [**FAIR-Play**](https://github.com/facebookresearch/FAIR-Play). 

2. Training. 
```bash
./train.sh
```
3. Ealuation.
```bash
./test.sh
```
A set of pretrained weights can be found at https://drive.google.com/drive/folders/1N7UMOZqNbFe_QXx4x_kKH02CQI8_DwPa?usp=sharing .


## Acknowledgement
We borrowed a lot of code from https://github.com/SheldonTsui/SepStereo_ECCV2020 and  https://github.com/facebookresearch/2.5D-Visual-Sound. Thanks for their great works. Please also cite their nice works if you use this code.

