
import torch
from torch import nn
import torchvision
from helpers.train import train_model
from helpers.plot import plot_result
import torch.optim as optim
from SegNet_GRU import SegNetGRU_Symmetric_columns_UltimateShare as SegNetGRU
from helpers.MyDataloaders import getLoaders
import numpy as np



def get_Dataloaders_dic(data_dir,train_val_ratio, input_size, shuffle,batch_size):
    Dataloaders_dic = {}
    # you can add more than 'data_C1','data_C2' (e.g., ['data_C1','data_C2','data_C3','data_C4']
    # It will be combined together and split into train/val according to the train_val_ratio
    dataloasers = getLoaders(['data_C1','data_C2'], input_size, data_dir=data_dir,
                             train_val_ratio=train_val_ratio, shuffle=shuffle, batch_size=batch_size)
    Dataloaders_dic['train'], Dataloaders_dic['val'] = dataloasers

    return Dataloaders_dic
def run():
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("device = ",device)

    data_dir = r"E:\Databases\EndoCV21\EndoCV21_sample"
    # Models to choose from [resnet18,resnet50,resnet152, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "SegNetGRU"
    # Number of classes in the dataset
    learning_rate = 0.01
    batch_size = 14
    input_size = (180, 225)
    train_val_ratio= 0.7 #i.e., 70% for training and 30% for validation
    shuffle= True # shuffle the images before split into train/val sets
    #batch_size = 16
    # Number of epochs to train for
    num_epochs = 100
    VGG_pretrained = True

    # Initialize the model for this run
    model_ft=SegNetGRU(VGG_pretrained=VGG_pretrained)
    # print(model_ft)
    # exit(0)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Print the model we just instantiated
    print("the used model is ",model_name)

    dataloaders_dict = get_Dataloaders_dic(data_dir,train_val_ratio, input_size, shuffle,batch_size)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
    # Train and evaluate
    model_ft, results_dic = train_model(model_ft, dataloaders_dict, criterion,
                                        optimizer_ft,device=device,
                                        num_epochs=num_epochs,input_size=input_size)
    plot_result(num_epochs=num_epochs,results_dic=results_dic)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run()