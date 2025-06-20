import torch
import numpy as np
from torch import optim
from torchmetrics.functional import precision, recall, f1_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from PIL import Image
import matplotlib.pyplot as plt
from config import *
import cv2

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=100):

    lr_lambda = lambda epoch: 1.0 - pow((epoch / decay_epoch), 0.9)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_scheduler.get_last_lr()[0]

    
def dice_score_metric(predictions, targets):
    dice_score = (2 * (predictions * targets).sum()) / ((predictions + targets).sum() + 1e-8)
    return dice_score

def IoU_metric(predictions, targets):
    intersection = (predictions*targets).sum()
    total = (predictions + targets).sum()
    union = total - intersection
    IoU = (intersection + 1e-8)/(union + 1e-8)
    return IoU

def get_metrics(pred, y, apply_sigmoid=False):
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    linear_y = torch.reshape(y>0, (-1,)).type(torch.int)
    linear_pred = torch.reshape(pred, (-1,)).type(torch.float)
    
    precision_val = precision(linear_pred, linear_y, task='binary', average='micro')
    recall_val = recall(linear_pred, linear_y, task='binary', average='micro')
    f1_score_val = f1_score(linear_pred,linear_y, task='binary', average='micro')
    return precision_val, recall_val, f1_score_val

def save_images(result, name, p, r, f1, IoU, dice):
    image = Image.open(VAL_IMG_DIR + name.replace('.png', '.jpg')).convert("RGB")
    mask = Image.open(VAL_MASK_DIR + name).convert("RGB")
    image = np.array(image)
    mask = np.array(mask)
    
    
    image_result =  Image.fromarray((result * 255).astype(np.uint8), mode='L')
    image_result.save(f"{IMG_PATH}test/{name}")
    
    # Create a mask for areas where result == 1 (successful segmentation)
    highlight_mask = np.zeros_like(image)
    highlight_mask[result == 1] = [255, 0, 0]  # Set highlighted areas to red
    
    # Combine the original image with the highlighted mask
    highlighted_image = cv2.addWeighted(image, 1, highlight_mask, 0.5, 0)
    
    # Define the figsize based on your preferred width and height
    figsize = (16, 4)  # Adjust the values as needed
    
    fig, axs = plt.subplots(1, 5, figsize=figsize)
    axs[0].imshow(image)
    axs[0].set_title("Input")
    axs[1].imshow(mask)
    axs[1].set_title("Ground Truth")
    axs[2].imshow(result, cmap='gray')
    axs[2].set_title("Output")
    
    # Center the text box within the fourth subplot
    text_box = axs[3].text(0.5, 0.5, f"Precision: {p:.2f}\nRecall: {r:.2f}\nF1 Score: {f1:.2f}\nIoU: {IoU:.2f}\nDice Coefficient: {dice:.2f}", 
                            color='white', fontsize=10, backgroundcolor='black', ha='center', va='center')
    axs[3].axis('off')  # Turn off axis for this subplot
    
    axs[4].imshow(highlighted_image)
    axs[4].set_title("Overlay")

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.2)  # Adjust wspace to your desired value

    fig.savefig(f"{IMG_PATH}val/{name}")
    
    plt.close(fig)

def erode_dilate(binary_mask, kernel_size=3, iterations=1):
    # Converter a máscara binária para um tipo de dados uint8
    binary_mask = np.uint8(binary_mask)

    # Definir o kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Erosão
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=iterations)

    # Dilatação
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=iterations)

    return dilated_mask

def get_metrics(pred, target, smooth=1e-8):
    #pred = erode_dilate_image(pred)
    #target = erode_dilate_image(target)
    pred_flat = np.reshape(pred,(-1))
    target_flat = np.reshape(target,(-1))
    
    #precision, recall, f1_score, _ = precision_recall_fscore_support(pred_flat, target_flat, average='binary')#, zero_division=0)
    precision, recall, f1_score = calculate_precision_recall_f1(pred_flat, target_flat)
    
    if (len(np.unique(pred_flat)) > 1) and (len(np.unique(target_flat)) > 1):
    # Calcular a métrica ROC AUC apenas se houver pelo menos duas classes presentes
        auc = roc_auc_score(target_flat, pred_flat)
    else:
        auc = 0
    
    intersection = (pred_flat*target_flat).sum()
    total = (pred_flat + target_flat).sum()
    union = total - intersection

    dice =  (2 * intersection + smooth) / (total + smooth)
        
    IoU = (intersection + smooth)/(union + smooth)
    
    
    
    return precision, recall, f1_score, IoU, dice, auc

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
    
def threshold_image(im,th):
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1
    return thresholded_im

def compute_otsu_criteria(im, th):
    thresholded_im = threshold_image(im,th)
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1
    if weight1 == 0 or weight0 == 0:
        return np.inf
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0
    return weight0 * var0 + weight1 * var1

def find_best_threshold(im):
    threshold_range = range(np.max(im)+1)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]
    return best_threshold


def get_otsu_threshold(pred_array):    
    pred_array = (255*pred_array).astype('uint8')
    best_threshold = find_best_threshold(pred_array)
    best_threshold = best_threshold/255
    
    return best_threshold

def erode_dilate_image(img):
    img /= img.max()
    kernel = np.ones((2,2),np.uint8)
    # Erosão da imagem
    imagem_erosion = cv2.erode(img, kernel, iterations = 1)
    # Dilatação da imagem
    imagem_dilation = cv2.dilate(imagem_erosion, kernel, iterations = 1)
    return imagem_dilation

def calculate_precision_recall_f1(y_pred, y_true, positive_class=1):
    tp = sum((y_pred == positive_class) & (y_true == positive_class))
    fp = sum((y_pred == positive_class) & (y_true != positive_class))
    fn = sum((y_pred != positive_class) & (y_true == positive_class))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score