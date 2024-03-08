import os
import click
import yaml
from data.dataset import build_dataloader
from Models.model1att2 import get_model
import torch
from torch.optim import AdamW
from tqdm import tqdm
import gc
import wandb
import numpy as np
import PIL
from torch.optim.lr_scheduler import LinearLR
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from torchmetrics.classification import BinaryF1Score

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "data/train_list.txt"
    if val_path is None:
        val_path = "data/val_list.txt"
    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path,'r') as f:
        val_list = f.readlines()
    
    return train_list, val_list

def load_data(train_path, val_path,batch_size, device,alpha):
    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list,batch_size=batch_size,num_workers=3,device=device,alpha=alpha)
    val_dataloader = build_dataloader(val_list,batch_size=batch_size,num_workers=3,device=device,validation=True,alpha=alpha)
    return train_dataloader,val_dataloader


def build_model():
    model = get_model()
    return model

#Input shape (dim1,dim2,channel)
def visualize_cuda_tensor(image_batch,pred_batch, gt_batch,name):
    image = np.moveaxis((image_batch.detach().cpu().numpy()[:,:,:] * 128 +128).astype(np.uint8),
                        0, -1)
    pred =  (pred_batch.detach().cpu().numpy()[0,:,:] * 255).astype(np.uint8)
    gt = (gt_batch.detach().cpu().numpy()[:,:] * 255).astype(np.uint8)

    #print("image.shape: ", image.shape)
    #print("pred.shape: ", pred.shape)
    #print("gt.shape: ", gt.shape)
    images = [PIL.Image.fromarray(image, mode="RGB"),
        PIL.Image.fromarray(pred).convert('RGB'),
        PIL.Image.fromarray(gt).convert('RGB')
        ]
    wandb.log({f"examples_{name}": [wandb.Image(image) for image in images]})

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
    #return -((2. * intersection + smooth) / (A_sum + B_sum + smooth))


def patch_validate(result_batch,gt_batch,eval_metric):# currently unused
    #step apply thresholding to both mask and predictions
    result_batch = torch.squeeze(result_batch)
    gt_batch = gt_batch.float().mean(dim=1).mean(dim=1) > 0.25
    result_batch = result_batch.float().mean(dim=1).mean(dim=1) > 0.25
    metric = eval_metric(result_batch,gt_batch)
    return metric

    #calculate average f1 score for batch

    #return average f1 score for batch

def validate(val_dataloader,model, i, pred_thres, device,patch_size=16):
    avg_acc = 0
    avg_prec = 0
    avg_rec = 0
    avg_F1 = 0
    avg_patch_f1 = 0
    avg_d_loss_score = 0
    counter = 0
    with torch.no_grad():
        for _, batch in enumerate(val_dataloader):
            image_batch = batch[0].to(device)
            gt_batch = batch[1].to(device)
            result = model(image_batch)
            acc, prec, rec, F1, patch_f1, d_loss_score = get_metrics(result,gt_batch,device)
            avg_acc += acc
            avg_prec += prec
            avg_rec += rec
            avg_F1 += F1
            avg_patch_f1 += patch_f1
            avg_d_loss_score += d_loss_score
            counter+= 1
        wandb.log({"avg_acc_validation": avg_acc/counter})
        wandb.log({"avg_prec_validation": avg_prec/counter})
        wandb.log({"avg_recall_validation": avg_rec/counter})
        wandb.log({"avg_F1_validation": avg_F1 /counter})
        #wandb.log({"avg_patch_F1_validation": avg_patch_f1/counter})
        wandb.log({"avg_dloss_validation": avg_d_loss_score/counter})
        visualize_cuda_tensor(image_batch[0],result[0],gt_batch[0],"validate")
    
        return avg_F1/counter

def saveModel(model, experiment_name,val=True):
    if not os.path.isdir(f"output/{experiment_name}/model_checkpoint"):
        os.mkdir(f"output/{experiment_name}/model_checkpoint")
    if val:
        torch.save(model.state_dict(),f"output/{experiment_name}/model_checkpoint/best_val.pth")
    else:
        torch.save(model.state_dict(),f"output/{experiment_name}/model_checkpoint/best.pth")

def initialize_optimizer_scheduler(model, num_epochs=100):
    lr = 3e-4
    optimizer = AdamW(lr=1.,params=model.parameters(),
                      weight_decay=1e-10)
    scheduler = LinearLR(start_factor = lr, 
                     end_factor = 1e-6,
                     last_epoch = -1,
                     total_iters = num_epochs,
                     optimizer = optimizer)
    
    return optimizer,scheduler

def initialize_wandb_config(wandb_config,config):
    wandb_config.num_epochs = config.get('num_epochs',100)
    wandb_config.device = config.get('device','cpu')
    wandb_config.batch_size = config.get('batch_size',10)
    wandb_config.val_freq = config.get('val_frequency',10)
    wandb_config.prediction_threshold = config.get('prediction_threshold',0.5)
    wandb_config.experiment_name = config.get('experiment_name')


def get_metrics(result_batch,gt_batch,device):
    #loss
    d_loss = dice_loss(result_batch, gt_batch)
    result_batch = result_batch > 0.5
    result_batch = torch.squeeze(result_batch)
    #Accuracy
    acc = BinaryAccuracy().to(device)
    acc_score = acc(result_batch,gt_batch)

    #Precision
    prec = BinaryPrecision().to(device)
    precision_score = prec(result_batch,gt_batch)

    #Recall
    rec = BinaryRecall().to(device)
    recall_score = rec(result_batch,gt_batch)

    #F1
    f1 = BinaryF1Score().to(device)
    F1_score = f1(result_batch,gt_batch)

    #Patch-F1
    patch_f1_score = 0
    #counter = 0
    #for j in range(0, result_batch.shape[2], 16):
     #   for i in range(0, result_batch.shape[1], 16):
      #      patch_score = patch_validate(result_batch[:,i:i + 16, j:j + 16], gt_batch[:,i:i + 16, j:j + 16],f1)
       #     patch_f1_score += patch_score
        #    counter += 1
    #patch_f1_score /= counter

    return acc_score, precision_score, recall_score, F1_score, patch_f1_score, d_loss

def train_one_epoch(model, optimizer,train_dataloader,device,plot,pred_thres): 
    avg_acc = 0
    avg_prec = 0
    avg_rec = 0
    avg_F1 = 0
    avg_patch_f1 = 0
    avg_d_loss_score = 0
    counter = 0
    for _, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        X, y = batch
        image_batch = X.to(device)
        gt_batch = y.to(device)
        result = model(image_batch)
        acc, prec, rec, F1, patch_f1, d_loss_score = get_metrics(result,gt_batch,device)
        avg_acc += acc
        avg_prec += prec
        avg_rec += rec
        avg_F1 += F1
        avg_patch_f1 += patch_f1
        avg_d_loss_score += d_loss_score
        counter+= 1
        #d_loss = dice_loss(result, gt_batch)

        #output, inverse_indices = torch.unique(
        #    torch.tensor(gt_batch, dtype=torch.long), sorted=True, return_inverse=True)
        #print(output)

        loss = d_loss_score 
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
    if plot:
        visualize_cuda_tensor(image_batch[0],result[0],gt_batch[0],"train")
    #Log metrics
    wandb.log({"avg_acc_training": avg_acc/counter})
    wandb.log({"avg_prec_training": avg_prec/counter})
    wandb.log({"avg_recall_training": avg_rec/counter})
    wandb.log({"avg_F1_training": avg_F1/counter})
    #wandb.log({"avg_patch_F1_training": avg_patch_f1/counter})
    wandb.log({"avg_dloss_training": avg_d_loss_score/counter})
    return avg_F1/counter

def train(config,model):
    num_epochs = config.get('num_epochs',100)
    train_path = config.get('train_data', None)
    val_path = config.get('val_dat', None)
    device = config.get('device','cpu')
    batch_size = config.get('batch_size',10)
    val_freq = config.get('val_frequency',5)
    train_freq = config.get('train_frequency',5)
    pred_thres = config.get('prediction_threshold',0.5)
    experiment_name = config.get('experiment_name')
    if not os.path.isdir(f"output/{experiment_name}"):
       if not os.path.isdir(f"output"):
           os.mkdir(f"output")
       os.mkdir(f"output/{experiment_name}")
    model.to(device)
    wandb_config = wandb.config
    initialize_wandb_config(wandb_config,config)
    optimizer, scheduler = initialize_optimizer_scheduler(model, num_epochs)
    best = - np.inf
    best_val = - np.inf
    for i in tqdm(range(num_epochs)):
        train_dataloader, val_dataloader = load_data(train_path,val_path,device=device,batch_size=batch_size,alpha=1)
        
        plot = (i % train_freq == 0)
        avg_metric = train_one_epoch(model, optimizer, train_dataloader,device,plot,pred_thres)
        if avg_metric > best:
            saveModel(model,experiment_name,False)
            best = avg_metric
        if scheduler is not None:# to not break the code if we have no lr scheduler
            scheduler.step()
        if i % val_freq == 0:
            model.eval()
            avg_val_metric = validate(val_dataloader,model,i,pred_thres, device)
            model.train()
            
            if avg_val_metric > best_val:
                saveModel(model,experiment_name)
                best_val = avg_val_metric
             

@click.command()
@click.option('-p', '--config_path', default='configs/config.yml',type=str)
def main(config_path):
    wandb.init(project="road_segmentation")
    config = yaml.safe_load(open(config_path))
    model = build_model()
    load_path = config.get('load_pth', "")
    if load_path != "":
        try:
            model.load_state_dict(torch.load(load_path))
            print("Restored weights from pretrained model.")
        except Exception as e:
            print(e)
            print("Error occurred when restoring weights.")
            val = ""
            while val != "y":
                val = input("Do you want to start training from scratch instead? (y/n):    ")
                if val == "n":
                    print("Aborting...")
                    exit()
    train(config, model)


if __name__ =="__main__":
    main()