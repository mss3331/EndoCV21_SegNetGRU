from PIL import Image
import torch
from torchvision import transforms
import cv2
import numpy as np
# can pass np array or path to image file
def detect(model, pathtest,train_img_size):
    if isinstance(pathtest, np.ndarray):
        img = Image.fromarray(pathtest)
    else:
        img = Image.open(pathtest)

    preprocess = transforms.Compose([transforms.Resize(train_img_size, 2),
                                     transforms.ToTensor(),
                                     ])
    Xtest = preprocess(img)



    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        model.to(device)
        Xtest = Xtest.to(device).float()
        ytest = model(Xtest.unsqueeze(0).float())
        ypos = ytest[0, 1, :, :].clone().detach().cpu().numpy()
        yneg = ytest[0, 0, :, :].clone().detach().cpu().numpy()
        ytest = ypos >= yneg

    mask = ytest.astype('float32')
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask = cv2.dilate(mask, kernel, iterations=2)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask