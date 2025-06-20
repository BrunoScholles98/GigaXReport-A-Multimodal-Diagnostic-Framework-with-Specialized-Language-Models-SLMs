import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from dataloader import get_loader,test_dataset, Ateroma_test_dataset
import torch.nn.functional as F
import numpy as np
from DC_UNet import DC_Unet
from run import validate
from config import *
import pandas as pd

DEVICE='cpu'
#NAME_CHECKPOINT = '/mnt/data/matheusvirgilio/gigasistemica/DC_UNET/runs/run 1 canal/checkpoint.pth.tar'
NAME_CHECKPOINT = '/mnt/data/matheusvirgilio/gigasistemica/DC_UNET/runs/run_com_menos_proc/checkpoint.pth.tar'
NAME_CHECKPOINT = '/mnt/data/matheusvirgilio/gigasistemica/DC_UNET/runs/2024-05-20 19:01:31.492590/checkpoint.pth.tar'
DATASET_DIR = '/mnt/data/matheusvirgilio/gigasistemica/datasets/giga_new/whole_dataset_revisited/new_dataset_validate_with_negatives/'
TRAIN_IMG_DIR = DATASET_DIR + "images/train/"
TRAIN_MASK_DIR = DATASET_DIR + "masks/train/"
VAL_IMG_DIR = DATASET_DIR + "images/val/"
VAL_MASK_DIR = DATASET_DIR + "masks/val/"

'''DATASET_DIR = '/mnt/data/matheusvirgilio/gigasistemica/datasets/giga_new/whole_dataset_revisited/demo_dataset/'
TRAIN_IMG_DIR = DATASET_DIR + "images/"
TRAIN_MASK_DIR = DATASET_DIR + "masks/"
VAL_IMG_DIR = DATASET_DIR + "images/"
VAL_MASK_DIR = DATASET_DIR + "masks/"'''


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    #model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

if __name__ == '__main__':
    model = DC_Unet(IN_CHANNELS).cpu()#.cuda(1)
    load_checkpoint(torch.load(NAME_CHECKPOINT, map_location=torch.device(DEVICE)), model)
    loader = Ateroma_test_dataset(VAL_IMG_DIR, VAL_MASK_DIR, TRAIN_SIZE)
    mean_loss, mean_precision, mean_recall, mean_f1_score, mean_IoU, mean_dice, mean_auc, metrics = validate(model, loader, save_test_images=True, apply_TTA=False, print_metrics=False, use_only_original_images=False)
    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df['name'] = metrics_df['name'].str.replace(".png", "")
    print("Média")
    print("validation/loss = ", mean_loss)
    print("validation/precision = ", mean_precision)
    print("validation/recall = ", mean_recall)
    print("validation/f1_score = ", mean_f1_score)
    print("validation/IoU_score = ", mean_IoU)
    print("validation/dice_score = ", mean_dice)
    print("validation/auc = ", mean_auc)
    
    print("Média das originais")
    metrics_df_1 = metrics_df.copy()
    colunas_para_media = [coluna for coluna in metrics_df_1['name'] if coluna.endswith(('R', 'L'))]
    media_por_coluna = metrics_df_1[metrics_df_1['name'].isin(colunas_para_media)].mean(numeric_only=True)
    print(media_por_coluna)
    
    print("Média das melhores")
    metrics_df_2 = metrics_df.copy()
    metrics_df_2['name'] = metrics_df_2.apply(lambda row: row['name'][:-1] if row['name'][-1] not in ['R', 'L'] else row['name'], axis=1)
    # Preservar apenas as linhas únicas, mantendo aquela com o maior valor na coluna 'iou'
    metrics_df_2 = metrics_df_2.sort_values('iou', ascending=False).drop_duplicates(subset='name')
    # Calcular a média por coluna
    media_por_coluna = metrics_df_2.mean(numeric_only=True)
    print(media_por_coluna)
    #casos_maiores_que_05 = a[a['recall'] > 0.8].shape[0]