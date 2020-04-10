# Pixelization of Natural Image via Cartoonization
## References
- CartoonGAN: Generative Adversarial Networks for Photo Cartoonization
- Deep Unsupervised Pixelization
## Requirement
- Python 3.5
- PIL
- Numpy
- Pytorch 0.4.0
- Torch
## Dataset
### Training Dataset (Deep Unsupervised Pixelization)
We collect 3000 portrait images and 900 pixel arts for traning our method. The folders named `trainA` and `trainB` contain the portrait images and pixel arts respectively. ([aligned celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [clip arts and pixel arts](https://drive.google.com/open?id=1qDXB5g0Cb0VwISXwnfeiehPHuTgxWhdG).

DUP's pretrained model is in `./checkpoints_pixelization`.  
CartoonGAN's pretained models are in `./cartoonGan/pretrained_model`.
### Testing Dataset
#### DUP
Create the folders `testA` and `testB` in a certain directory. Note that `testA` and `testB` contain the cartoon arts to be pixelized and pixel arts to be depixelized respectively.
#### CartoonGAN
Input natural images - portrait [aligned celeba](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
## Training
* To train a model (DUP):
``` bash
python3 ./train.py --dataroot ./samples --resize_or_crop resize_and_crop --gpu_ids 0
```
## Testing
* After training, all models have been saved in the directory `./checkpoints_pixelization/`.
* To test a model (DUP):
``` bash
python3 ./test.py --dataroot ./samples --resize_or_crop resize_and_crop --gpu_ids 0 --how_many 1 --which_epoch 150 --loadSize 256
```
* To test a model (Pixelization of Natural Image via Cartoonization)
``` bash
python ./main.py --loadSize 320
```
* Note : 
`main.py` will cartoonized portrait images in `./input/natural_input` and output to `./input/cartoon_input/testA`  
Then `testA` and `testB` in `./input/cartoon_input` will be pixelized and depixelized respectivelty.  
More testing flags in the file `./options/base_options.py`.  
All testing results will be shown in the directory `./results_pixelization/`.
