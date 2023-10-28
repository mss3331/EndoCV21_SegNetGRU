import matplotlib.pyplot as plt
import numpy as np
def plot_result(num_epochs,results_dic):

    # Plot the training curves of validation accuracy & loss
    val_acc = [h.cpu().numpy() for h in results_dic["val_acc_history"]]
    val_loss = [h for h in results_dic["val_loss_history"]]
    train_acc = [h.cpu().numpy() for h in results_dic["train_acc_history"]]
    train_loss = [h for h in results_dic["train_loss_history"]]


    plt.title("Accuracy & Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,num_epochs+1),val_acc,"b",label="Validation Acc")
    plt.plot(range(1,num_epochs+1),val_loss,"--b",label="Validation Loss")
    plt.plot(range(1,num_epochs+1),train_acc,"k",label="Train Acc")
    plt.plot(range(1,num_epochs+1),train_loss,"--k",label="Train Loss")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 10))
    plt.yticks(np.arange(0,1.5, 0.1))
    plt.legend()
    plt.grid(True)
    plt.show()