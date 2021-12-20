# EndoCV21_SegNetGRU
This is a pytorch implementation code for the proposed model 'SegNetGRU'. It is a segmentation model employed to segment polyps in EndoCV21 dataset.
For more information read the following paper: http://ceur-ws.org/Vol-2886/paper6.pdf.

**This repo contains the following:
1. **checkpoints (Folder)**: should contains the model's checkpoint
2. **SegNet_GRU.py (model implementation)**: contains Pytorch code of my model
3. **Detection.py (helper)**: the actual segmentation is conducted here for a single image
4. **endocv2021_inference-seg.py (main)**: this is the main code that reads images and store the masks in the respected folders (i.e, according to the EndoCV21 challenge rules)
