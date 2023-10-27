import torch
import numpy as np

def TP_TN_FP_FN(true, pred):
    minus = true - pred
    FP = (minus==-1).float().sum(dim=1)
    FN = (minus==1).float().sum(dim=1)
    TP_plus_TN = (minus==0).float().sum(dim=1)

    multi = true * pred
    TP = multi.sum(dim=1)
    TN = TP_plus_TN - TP
    return TP, TN, FP, FN

def calculate_metrics_torch(true, pred, ROI='polyp',metrics=None, reduction=None,
                            cloned_detached=True):
    '''The input are tensors of shape (batch, C, H, W)'''

    batch_size = pred.shape[0]
    # (batch, C, H, W)->(batch,HW)
    if cloned_detached:
        true = true.argmax(dim=1).view(batch_size, -1).float()
        pred = pred.argmax(dim=1).view(batch_size, -1).float()
    else:
        true = true.clone().detach().argmax(dim=1).view(batch_size, -1).float()
        pred = pred.clone().detach().argmax(dim=1).view(batch_size, -1).float()

    #-------------------------------------------
    if metrics== None:
        metrics = 'accuracy', 'jaccard', 'dic', 'recall', 'precision'
    elif type(metrics)==str:
        metrics = [metrics]

    #-------------------------------------------
    if ROI=='polyp':
        pass
    elif ROI=='background':
        true = 1- true
        pred = 1- pred
    true.requires_grad = False
    pred.requires_grad = False
    #--------------------------------------------
    TP, TN, FP, FN =TP_TN_FP_FN(true,pred)
    #--------------------------------------------
    results = []
    for metric in metrics:
        result = 0
        if metric=='jaccard':
            iou = TP/(TP + FN + FP)
            result = iou
        elif metric=='accuracy':
            acc = (TP+TN)/(TP + FN + FP + TN)
            result = acc
        elif metric=='dic':
            dic = 2*TP/(2*TP + FP + FN)
            result = dic
        elif metric=='recall':
            recall = TP/(TP+FN)
            result = recall
        elif metric == 'precision':
            TP_FP = (TP + FP)
            TP_FP[TP_FP==0]+=1
            prec = TP / TP_FP
            result = prec
        else:
            continue

        if reduction=='mean':
            result = torch.mean(result).cpu().numpy().round(5)
        else:
            result = result.cpu().numpy().round(5)

        results.append(result)

    results = np.array(results)
    results_iou_dic = {metric:results[index] for index, metric in enumerate(metrics)}

    if len(results)==1:
        results_iou_dic = results[0]


    return results_iou_dic