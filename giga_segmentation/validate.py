import torch
import numpy as np
from utils import save_images, get_metrics, adjust_lr, save_checkpoint, get_otsu_threshold, erode_dilate_image
import torch.nn.functional as F
from loss import structure_loss

def validate(model, loader, save_test_images=False, print_metrics=False):
    model.eval()
    
    loader.index = 0
    loader_size = loader.size
    
    metrics = {'name' : [], 'area' : [], 'loss' : [], 'precision' : [], 'recall' : [], 'f1 score' : [], 'iou' : [], 'dice score' : []}
    mean_loss = 0
    mean_precision = 0
    mean_recall = 0
    mean_f1_score = 0
    mean_IoU = 0
    mean_dice = 0
    mean_auc = 0
    
    for i in range(loader_size):
        image, target, name = loader.load_data()
        
        image = image.cuda()
        target_array = np.asarray(target, np.float32)
        #target_array = erode_dilate_image(target_array)
        target_array /= (target_array.max() + 1e-8)
        
        prediction = model(image)
        prediction = F.interpolate(prediction, size=target_array.shape, mode='bilinear', align_corners=False)
        
        #calcular a loss

        loss = structure_loss(prediction.cpu(), torch.from_numpy(target_array).unsqueeze(0).unsqueeze(0)).item()
        
        prediction = prediction.sigmoid().data.cpu().numpy().squeeze()
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
        
        #if save_test_images:
        #    save_images(prediction, name)
        
        #threshold = get_otsu_threshold(prediction)
        threshold = 0.5
        prediction[prediction >= threshold] = 1.0
        prediction[prediction < threshold] = 0
        
        p, r, f1, IoU, dice, auc = get_metrics(prediction, target_array)
        
        if save_test_images:
            save_images(prediction, name, p, r, f1, IoU, dice)
        
        mean_loss += loss
        mean_precision += p
        mean_recall += r
        mean_f1_score += f1
        mean_IoU += IoU
        mean_dice += dice
        mean_auc += auc
        
        metrics['name'] += [name]
        metrics['area'] += [target_array.sum()]
        metrics['loss'] += [loss]
        metrics['precision'] += [p]
        metrics['recall'] += [r]
        metrics['f1 score'] += [f1]
        metrics['iou'] += [IoU]
        metrics['dice score'] += [dice]
        
        if(print_metrics):
            print(name)
            print("validation/precision = ", p)
            print("validation/recall = ", r)
            print("validation/f1_score = ", f1)
            print("validation/IoU_score = ", IoU)
            print("validation/dice_score = ", dice)
    
    mean_loss = mean_loss/loader_size
    mean_precision = mean_precision/loader_size
    mean_recall = mean_recall/loader_size
    mean_f1_score = mean_f1_score/loader_size
    mean_IoU = mean_IoU/loader_size
    mean_dice = mean_dice/loader_size
    mean_auc = mean_auc/loader_size

    return mean_loss, mean_precision, mean_recall, mean_f1_score, mean_IoU, mean_dice, mean_auc, metrics