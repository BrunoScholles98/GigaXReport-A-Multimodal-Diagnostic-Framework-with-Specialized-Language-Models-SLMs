import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import os
import datetime
from tqdm import tqdm
from DC_UNet import DC_Unet
from dataloader import get_loader, test_dataset, Ateroma_test_dataset
#from ateroma_dataloader import get_loader, Ateroma_test_dataset, Ateroma_test_dataset
from loss import structure_loss
from utils import adjust_lr, save_checkpoint
from validate import validate

#Paths
#DATASET_DIR = "/mnt/data/matheusvirgilio/gigasistemica/datasets/giga_new/dataset_corner_without_negatives_aug/"
#DATASET_DIR = '/mnt/data/matheusvirgilio/gigasistemica/datasets/giga_new/whole_dataset_revisited/new_dataset_validate_with_negatives/'
DATASET_DIR = '/mnt/nas/matheusvirgilio/gigasistemica/datasets/giga_new/final_dataset/folds/fold_1/'
TRAIN_IMG_DIR = DATASET_DIR + "images/train/"
TRAIN_MASK_DIR = DATASET_DIR + "masks/train/"
VAL_IMG_DIR = DATASET_DIR + "images/val/"
VAL_MASK_DIR = DATASET_DIR + "masks/val/"
RUNS_DIR = "/mnt/nas/matheusvirgilio/gigasistemica/DC_UNET/runs/"

#Model Features
IN_CHANNELS = 1
TRAIN_SIZE = 512
AUGMENTATION = True

#Hyperparemeters
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_WORKERS = 0

NAME_CHECKPOINT = '/mnt/data/matheusvirgilio/gigasistemica/DC_UNET/runs/run_mascaras_refeitas/checkpoint1.pth.tar'
DEVICE = 'cuda'
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    #model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

def create_run_dir_with_description(runs_directory):
    RUN_NAME =  str(datetime.datetime.now()) + "/"
    RUN_DIR = runs_directory + RUN_NAME
    os.makedirs(RUN_DIR)
    IMGS_SAVED_DIR = RUN_DIR + "saved_images/"
    os.makedirs(IMGS_SAVED_DIR)
    os.makedirs(IMGS_SAVED_DIR + "val/")
    os.makedirs(IMGS_SAVED_DIR + "test/")
    return RUN_DIR

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
        
        loss = structure_loss(predictions, targets)              
        loss_value = loss.item()
        mean_loss += loss_value
        loop.set_postfix(loss = loss_value) # update tqdm loop
        loss.backward()
        optimizer.step()
    
    mean_loss = mean_loss/len_loader
    return mean_loss

if __name__ == '__main__':
    run_directory = create_run_dir_with_description(RUNS_DIR)
    checkpoint = run_directory + "checkpoint.pth.tar"
    writer = SummaryWriter(log_dir= run_directory + "plot/")
    # ---- build models ----
    torch.cuda.set_device(1)  # set your gpu device
    
    model = DC_Unet(in_channels=IN_CHANNELS).cuda()
    #load_checkpoint(torch.load(NAME_CHECKPOINT, map_location=torch.device(DEVICE)), model)
    
    params = model.parameters()
    optimizer = torch.optim.Adam(params, LEARNING_RATE)
        

    train_loader = get_loader(TRAIN_IMG_DIR, TRAIN_MASK_DIR, 
                              batchsize= BATCH_SIZE, 
                              trainsize=TRAIN_SIZE, 
                              num_workers=NUM_WORKERS, 
                              augmentation = AUGMENTATION)
    
    val_loader = Ateroma_test_dataset(VAL_IMG_DIR, VAL_MASK_DIR, TRAIN_SIZE)
    #val_loader = test_dataset(VAL_IMG_DIR, VAL_MASK_DIR, TRAIN_SIZE)
    total_step = len(train_loader)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        adjust_lr(optimizer, LEARNING_RATE, epoch, 0.1, 100)
        mean_loss = train(model, train_loader, optimizer, epoch)
        
        # save model
        checkpoint_dict = {"state_dict" : model.state_dict(), "optimizer" : optimizer.state_dict()}
        save_checkpoint(checkpoint_dict, filename=checkpoint)
        
        mean_loss_val, mean_precision, mean_recall, mean_f1_score, mean_IoU, mean_dice, _, _ = validate(model, val_loader, save_test_images=False)
        writer.add_scalar('training/loss', mean_loss, epoch)
        writer.add_scalar('validation/loss', mean_loss_val, epoch)
        writer.add_scalar('validation/precision', mean_precision, epoch)
        writer.add_scalar('validation/recall', mean_recall, epoch)
        writer.add_scalar('validation/f1_score', mean_f1_score, epoch)
        writer.add_scalar('validation/IoU', mean_IoU, epoch)
        writer.add_scalar('validation/score', mean_dice, epoch)