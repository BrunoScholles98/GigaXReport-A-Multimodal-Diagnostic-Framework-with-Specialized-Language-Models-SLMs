import albumentations as A
from albumentations.pytorch import transforms as A_torch
from albumentations.augmentations.functional import _maybe_process_in_chunks, preserve_shape
import torch
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
import cv2
import albumentations as A
import random
import torch.nn.functional as F


def apply_TTA(image):
    image_array = image.cpu().numpy().copy()
    image_array = image_array.squeeze()  # Remover as dimensões extras (1, 1)
    #image_array = image_array.transpose(2, 3, 0, 1)  # Mudar a ordem dos eixos para (512, 512, 1)

    #flip_image = torch.Tensor(image.cpu().numpy()[:,:,:,::-1].copy())
    scale_limit = random.uniform(0, 0.1)
    rotate_limit = random.randint(0, 5)
    translate_px = random.randint(-25, 25)
    
    transformation1 = A.Compose(A.Affine(translate_px={"x": translate_px, "y": translate_px}, p=1))
    transformation2 = A.Compose(A.ShiftScaleRotate(shift_limit=0, scale_limit=(scale_limit, scale_limit), rotate_limit=(rotate_limit, rotate_limit), p=1))
    
    
    image_translated = transformation1(image=image_array)
    image_scaled_rotated = transformation2(image=image_array)
    image_fliped_scaled_rotated = transformation2(image=image_array[:,::-1])
    
    image_translated = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image_translated['image']), dim=0), dim=0)
    image_scaled_rotated = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image_scaled_rotated['image']), dim=0), dim=0)
    image_fliped_scaled_rotated = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(image_fliped_scaled_rotated['image']), dim=0), dim=0)
    
    return image_translated, image_scaled_rotated, image_fliped_scaled_rotated, (scale_limit, rotate_limit, translate_px)

def revert_TTA(mask, mask_translated, mask_scaled_rotated, mask_fliped_scaled_rotated, parameters):
    scale_limit, rotate_limit, translate_px = -parameters[0], -parameters[1], -parameters[2]
    transformation1 = A.Compose(A.Affine(translate_px={"x": translate_px, "y": translate_px}, p=1))
    transformation2 = A.Compose(A.ShiftScaleRotate(shift_limit=0, scale_limit=(scale_limit, scale_limit), rotate_limit=(rotate_limit, rotate_limit), p=1))
    
    cv2.imwrite('/mnt/data/matheusvirgilio/gigasistemica/dataset_generation/demonstration images/imagem_salva_opencv.png', 255*mask)
    cv2.imwrite('/mnt/data/matheusvirgilio/gigasistemica/dataset_generation/demonstration images/imagem_salva_opencv1.png', 255*mask_translated)
    cv2.imwrite('/mnt/data/matheusvirgilio/gigasistemica/dataset_generation/demonstration images/imagem_salva_opencv2.png', 255*mask_scaled_rotated)
    cv2.imwrite('/mnt/data/matheusvirgilio/gigasistemica/dataset_generation/demonstration images/imagem_salva_opencv3.png', 255*mask_fliped_scaled_rotated)
    
    mask1 = transformation1(image=mask_translated)
    mask1 = mask1['image']
    mask2 = transformation2(image=mask_scaled_rotated)
    mask2 = mask2['image']
    mask3 = transformation2(image=mask_fliped_scaled_rotated)
    mask3 = mask3['image'][:,::-1]
    
    cv2.imwrite('/mnt/data/matheusvirgilio/gigasistemica/dataset_generation/demonstration images/imagem_salva_opencv.png', 255*mask)
    cv2.imwrite('/mnt/data/matheusvirgilio/gigasistemica/dataset_generation/demonstration images/imagem_salva_opencv1.png', 255*mask1)
    cv2.imwrite('/mnt/data/matheusvirgilio/gigasistemica/dataset_generation/demonstration images/imagem_salva_opencv2.png', 255*mask2)
    cv2.imwrite('/mnt/data/matheusvirgilio/gigasistemica/dataset_generation/demonstration images/imagem_salva_opencv3.png', 255*mask3)
    aux = get_final_bin_mask(mask, mask1, mask2, mask3)
    cv2.imwrite('/mnt/data/matheusvirgilio/gigasistemica/dataset_generation/demonstration images/imagem_salva_opencv3.png', 255*aux)
    return aux

def get_final_bin_mask(orig_bin_mask, mask1, mask2, mask3):
    sum_bin_mask = orig_bin_mask + mask1 + mask2 + mask3 #+ v_bin_mask
    final_bin_mask = sum_bin_mask >= 2
    return final_bin_mask

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

def TTA_data(data, DEVICE):
    flip_data, filter_data = apply_TTA(data)
    flip_data = flip_data.to(device=DEVICE)
    filter_data = filter_data.to(device=DEVICE)
    return torch.cat([data, flip_data, filter_data], axis=0)