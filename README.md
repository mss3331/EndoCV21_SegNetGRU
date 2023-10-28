# EndoCV21_SegNetGRU
This is a pytorch implementation code for the proposed model 'SegNetGRU'. It is a segmentation model employed to segment polyps in EndoCV21 dataset.
For more information read the following paper: http://ceur-ws.org/Vol-2886/paper6.pdf.

## This repo contains the following:
1. **checkpoints (Folder)**: should contains the model's checkpoint after conducting training (run main.py)
2. **helper (Folder)**: contains helper python files
3. **main (training code)**: contain all hyperparameters and intiate training. Best models weight are saved in checkpoints folder
4. **SegNet_GRU.py (model implementation)**: contains Pytorch code of SegNetGRU
6. **endocv2021_inference-seg.py (inference code)**: this is the main code for inference after conducting training (i.e., run main). The code reads images from testfolders and store the masks in the respected folders (i.e, according to the EndoCV21 challenge rules)
