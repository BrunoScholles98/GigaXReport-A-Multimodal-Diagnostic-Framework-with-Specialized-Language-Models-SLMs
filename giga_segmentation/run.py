import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from config import *
import numpy as np
from torch.autograd import Variable

import torch.nn.functional as F

from DC_UNet import DC_Unet

from dataloader import get_loader, test_dataset, Ateroma_test_dataset
from torchvision.ops import sigmoid_focal_loss

#from ateroma_dataloader import get_loader, Ateroma_test_dataset, Ateroma_test_dataset


from loss import structure_loss
from config import *
from utils import save_images, get_metrics, adjust_lr, save_checkpoint, get_otsu_threshold, erode_dilate_image, calculate_precision_recall_f1
from TTA import apply_TTA, revert_TTA

import cv2

writer = SummaryWriter(log_dir= RUN_DIR + "plot/")



def TTA(model, image, target, prediction, threshold = 0.5):
    image1, image2, image3, parameters = apply_TTA(image)
    prediction1 = model(image1)
    prediction1 = F.interpolate(prediction1, size=target.shape, mode='bilinear', align_corners=False)
    prediction2 = model(image2)
    prediction2 = F.interpolate(prediction2, size=target.shape, mode='bilinear', align_corners=False)
    prediction3 = model(image3)
    prediction3 = F.interpolate(prediction3, size=target.shape, mode='bilinear', align_corners=False)
    prediction1 = prediction1.sigmoid().data.cpu().numpy().squeeze()
    prediction2 = prediction2.sigmoid().data.cpu().numpy().squeeze()
    prediction3 = prediction3.sigmoid().data.cpu().numpy().squeeze()
    prediction1 = (prediction1 - prediction1.min()) / (prediction1.max() - prediction1.min() + 1e-8)
    prediction2 = (prediction2 - prediction2.min()) / (prediction2.max() - prediction2.min() + 1e-8)
    prediction3 = (prediction3 - prediction3.min()) / (prediction3.max() - prediction3.min() + 1e-8)
    prediction1[prediction1 >= threshold] = 1.0
    prediction1[prediction1 < threshold] = 0
    prediction2[prediction2 >= threshold] = 1.0
    prediction2[prediction2 < threshold] = 0
    prediction3[prediction3 >= threshold] = 1.0
    prediction3[prediction3 < threshold] = 0
    return revert_TTA(prediction, prediction1, prediction2, prediction3, parameters)
            
def validate(model, loader, save_test_images=False, print_metrics=False, use_only_original_images=False, apply_TTA=False):
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
    n = 0
    
    for i in range(loader_size):
        image, target, name = loader.load_data()
        
        aux = name.replace('.png', '')[-1]
        if((aux == 'R' or aux == 'L') or not (use_only_original_images or apply_TTA)):
            n += 1
            #image = image.to(device)
            
            image = image.cpu()
            target_array = np.asarray(target, np.float32)
            #target_array = erode_dilate_image(target_array)
            target_array /= (target_array.max() + 1e-8)
            
            prediction = model(image)
            prediction = F.interpolate(prediction, size=target_array.shape, mode='bilinear', align_corners=False)
            
            #calcular a loss
            if LOSS_FUNCTION == 'Focal Loss':
                loss = sigmoid_focal_loss(prediction.cpu(), torch.from_numpy(target_array).unsqueeze(0).unsqueeze(0), reduction='mean', gamma=1).item()
            if LOSS_FUNCTION == 'IoU':
                loss = structure_loss(prediction.cpu(), torch.from_numpy(target_array).unsqueeze(0).unsqueeze(0)).item()
            
            prediction = prediction.sigmoid().data.cpu().numpy().squeeze()
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            
            #if save_test_images:
            #    save_images(prediction, name)
            
            #threshold = get_otsu_threshold(prediction)
            threshold = 0.5
            prediction[prediction >= threshold] = 1.0
            prediction[prediction < threshold] = 0
            
            if (apply_TTA):
                prediction = TTA(model, image, target_array, prediction)
            
            p, r, f1, IoU, dice, auc = get_metrics(prediction, target_array)
            
            if(np.all(target_array == 0)):
                pred_flat = np.reshape(prediction,(-1))
                target_flat = np.reshape(target_array,(-1))
                p, r, f1 = calculate_precision_recall_f1(pred_flat, target_flat, positive_class=0)
            
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
    
    mean_loss = mean_loss/n
    mean_precision = mean_precision/n
    mean_recall = mean_recall/n
    mean_f1_score = mean_f1_score/n
    mean_IoU = mean_IoU/n
    mean_dice = mean_dice/n
    mean_auc = mean_auc/n

    return mean_loss, mean_precision, mean_recall, mean_f1_score, mean_IoU, mean_dice, mean_auc, metrics

def train(model, loader, optimizer, epoch):
    model.train()
    
    mean_loss = 0
    
    loop = tqdm(loader)
    len_loader = len(loader)
    
    #with torch.enable_grad()
    for batch_idx, (data, targets) in enumerate(loop):
        optimizer.zero_grad()
        #data = data.to(device=DEVICE)
        #targets = targets.float().unsqueeze(1).to(device=DEVICE)
        data = Variable(data).cuda()
        targets = Variable(targets).cuda()
        predictions = model(data)
        
        if LOSS_FUNCTION == 'Focal Loss':
            loss = sigmoid_focal_loss(predictions, targets, reduction='mean', gamma=1)
        if LOSS_FUNCTION == 'IoU':
            loss = structure_loss(predictions, targets)
              
        loss_value = loss.item()
        mean_loss += loss_value
        loop.set_postfix(loss = loss_value) # update tqdm loop
        loss.backward()
        # clip_gradient(optimizer, opt.clip)
        optimizer.step()
    
    mean_loss = mean_loss/len_loader
    return mean_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=NUM_EPOCHS, help='epoch number')
    
    parser.add_argument('--lr', type=float,
                        default=LEARNING_RATE, help='learning rate')
    
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    
    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    
    parser.add_argument('--batchsize', type=int,
                        default=BATCH_SIZE, help='training batch size')
    
    parser.add_argument('--trainsize', type=int,
                        default=TRAIN_SIZE, help='training dataset size')
    
    parser.add_argument('--train_path', type=str,
                        default=TRAIN_IMG_DIR, help='path to train dataset')
    
    parser.add_argument('--test_path', type=str,
                        default=VAL_IMG_DIR , help='path to testing Kvasir dataset')
    
    parser.add_argument('--train_save', type=str,
                        default='dcunet-best')
    
    opt = parser.parse_args()
    
        # ---- build models ----
    torch.cuda.set_device(1)  # set your gpu device
    model = DC_Unet(in_channels=IN_CHANNELS).cuda()
    params = model.parameters()
    
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, opt.lr)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay = 1e-4, momentum = 0.9)
        
    image_root = TRAIN_IMG_DIR
    gt_root = TRAIN_MASK_DIR

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=NUM_WORKERS, augmentation = opt.augmentation)
    val_loader = Ateroma_test_dataset(VAL_IMG_DIR, VAL_MASK_DIR, TRAIN_SIZE)
    #val_loader = test_dataset(VAL_IMG_DIR, VAL_MASK_DIR, TRAIN_SIZE)
    total_step = len(train_loader)
    
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 100)
        mean_loss = train(model, train_loader, optimizer, epoch)
        
        # save model
        checkpoint = {"state_dict" : model.state_dict(), "optimizer" : optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=NAME_CHECKPOINT)
        
        mean_loss_val, mean_precision, mean_recall, mean_f1_score, mean_IoU, mean_dice, _, _ = validate(model, val_loader, save_test_images=True)
        writer.add_scalar('training/loss', mean_loss, epoch)
        writer.add_scalar('validation/loss', mean_loss_val, epoch)
        writer.add_scalar('validation/precision', mean_precision, epoch)
        writer.add_scalar('validation/recall', mean_recall, epoch)
        writer.add_scalar('validation/f1_score', mean_f1_score, epoch)
        writer.add_scalar('validation/IoU', mean_IoU, epoch)
        writer.add_scalar('validation/score', mean_dice, epoch)