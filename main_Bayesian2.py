import os
import click
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from data.dataset import build_dataloader, ImageList
from Models.model1_BAYESIAN import get_model_two_step
import torch
from torch.optim import AdamW
from tqdm import tqdm
import gc
import wandb
import numpy as np
import PIL
from torch.optim.lr_scheduler import LinearLR
import re
from main_Bayesian import b_validate, visualize_cuda_tensor
from main import get_metrics


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


def load_data(train_path, val_path, batch_size, device, alpha, has_uncert=False):
    train_list, val_list = get_data_path_list(train_path, val_path)
    train_dataloader = build_dataloader(train_list, batch_size=batch_size, num_workers=4, device=device, alpha=alpha,
                                        has_uncert=has_uncert)
    val_dataloader = build_dataloader(val_list, batch_size=batch_size, num_workers=4, device=device, validation=True,
                                      alpha=alpha, has_uncert=has_uncert)
    return train_dataloader, val_dataloader


def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def saveModel(model, experiment_name,val=True):
    if not os.path.isdir(f"output/{experiment_name}/model_checkpoint"):
        os.mkdir(f"output/{experiment_name}/model_checkpoint")
    if val:
        torch.save(model.state_dict(),f"output/{experiment_name}/model_checkpoint/best_val.pth")
    else:
        torch.save(model.state_dict(),f"output/{experiment_name}/model_checkpoint/best.pth")


def initialize_optimizer_scheduler(model, num_epochs=100):
    lr = 3e-4
    optimizer = AdamW(lr=1., params=model.parameters(),
                      weight_decay=1e-10)
    scheduler = LinearLR(start_factor=lr,
                         end_factor=1e-6,
                         last_epoch=-1,
                         total_iters=num_epochs,
                         optimizer=optimizer)

    return optimizer, scheduler


def initialize_wandb_config(wandb_config, config):
    wandb_config.num_epochs = config.get('num_epochs', 100)
    wandb_config.device = config.get('device', 'cpu')
    wandb_config.batch_size = config.get('batch_size', 10)
    wandb_config.val_freq = config.get('val_frequency', 10)
    wandb_config.prediction_threshold = config.get('prediction_threshold', 0.5)
    wandb_config.experiment_name = config.get('experiment_name')


def train_one_epoch(model, optimizer, train_dataloader, device, plot, pred_thres):
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

        # output, inverse_indices = torch.unique(
        #    torch.tensor(gt_batch, dtype=torch.long), sorted=True, return_inverse=True)
        # print(output)
        loss = d_loss_score
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
        wandb.log({"loss": loss})
    if plot:
        visualize_cuda_tensor(image_batch[0], result[0], gt_batch[0], "train")
    #Log metrics
    wandb.log({"avg_acc_training": avg_acc/counter})
    wandb.log({"avg_prec_training": avg_prec/counter})
    wandb.log({"avg_recall_training": avg_rec/counter})
    wandb.log({"avg_F1_training": avg_F1/counter})
    #wandb.log({"avg_patch_F1_training": avg_patch_f1/counter})
    wandb.log({"avg_dloss_training": avg_d_loss_score/counter})
    return avg_F1/counter


def train(config, model, model2):
    num_epochs = config.get('num_epochs', 100)
    num_epochs2 = config.get('num_epochs2', 100)
    train_path = config.get('train_data', None)
    val_path = config.get('val_dat', None)
    device = config.get('device', 'cpu')
    batch_size = config.get('batch_size', 10)
    val_freq = config.get('val_frequency', 5)
    train_freq = config.get('train_frequency', 5)
    pred_thres = config.get('prediction_threshold', 0.5)
    experiment_name = config.get('experiment_name')
    if not os.path.isdir(f"output/{experiment_name}"):
        if not os.path.isdir(f"output"):
            os.mkdir(f"output")
        os.mkdir(f"output/{experiment_name}")
    model.to(device)
    wandb_config = wandb.config
    initialize_wandb_config(wandb_config, config)

    # Train first u-net
    optimizer, scheduler = initialize_optimizer_scheduler(model, num_epochs)
    best = - np.inf
    best_val = - np.inf
    for i in tqdm(range(num_epochs)):
        train_dataloader, val_dataloader = load_data(train_path, val_path, device=device, batch_size=batch_size,
                                                     alpha=1)

        plot = (i % train_freq == 0)
        avg_f1 = train_one_epoch(model, optimizer, train_dataloader, device, plot, pred_thres)
        if avg_f1 > best:
            saveModel(model,experiment_name,False)
            best = avg_f1
        if scheduler is not None:  # to not break the code if we have no lr scheduler
            scheduler.step()
        if i % val_freq == 0:
            avg_val_metric = b_validate(val_dataloader, model, i, pred_thres, device)
            if avg_val_metric > best_val:
                saveModel(model,experiment_name)
                best_val = avg_val_metric

    calc_uncert = True
    # Compute uncertainties
    #(just take the last model)
    # its more important that for the 2. model the best one is chosen
    print("calculate uncertainties")
    if calc_uncert:
        if not os.path.isdir(f"data/training/uncertainty"):
            os.mkdir(f"data/training/uncertainty")
        if not os.path.isdir(f"data/valid/uncertainty"):
            os.mkdir(f"data/valid/uncertainty")
        train_list, val_list = get_data_path_list(train_path, val_path)
        train_list = [entry.split('|')[0] for entry in train_list]
        val_list = [entry.split('|')[0] for entry in val_list]
        file_list = train_list + val_list
        img_loader = DataLoader(ImageList(file_list))
        with torch.no_grad():
            for i, batch in enumerate(img_loader):
                img_batch = batch[0].to(device)
                results = []
                for _ in range(20):
                    pred = model(img_batch)
                    results.append(pred)

                result = torch.var(torch.stack(results), dim=0)
                # max var of binomial = .25
                result *= 100 # [0, 250] 
                result = torch.round(result)

                # result = torch.where(result > 0.025, 1, 0)
                result = (result[0].cpu().numpy()[0, :, :]).astype(np.uint8)
                img = Image.fromarray(result, mode='L')
                img.save('data/' + file_list[i].replace('images', 'uncertainty'))

    # Train second u-net
    model2.to(device)
    optimizer, scheduler = initialize_optimizer_scheduler(model2, num_epochs2)
    best = - np.inf
    best_val = - np.inf
    print("train 2. model")
    for i in tqdm(range(num_epochs2)):
        train_dataloader, val_dataloader = load_data(train_path, val_path, device=device, batch_size=batch_size,
                                                     alpha=1, has_uncert=True)

        plot = (i % train_freq == 0)
        avg_f1 = train_one_epoch(model2, optimizer, train_dataloader, device, plot, pred_thres)
        if avg_f1 > best:
            saveModel(model,experiment_name,False)
            best = avg_f1
        if scheduler is not None:  # to not break the code if we have no lr scheduler
            scheduler.step()
        if i % val_freq == 0:
            avg_val_metric = b_validate(val_dataloader, model2, i, pred_thres, device)
            if avg_val_metric > best_val:
                saveModel(model,experiment_name)
                best_val = avg_val_metric


@click.command()
@click.option('-p', '--config_path', default='configs/config.yml', type=str)
def main(config_path):
    wandb.init(project="road_segmentation")
    config = yaml.safe_load(open(config_path))
    model, model2 = get_model_two_step()
    load_path = config.get('load_pth', "")
    if "pth" in load_path:
        load_path = os.path.dirname(os.path.normpath(load_path))

    if load_path != "":
        try:
            model.load_state_dict(torch.load(load_path + '/model1.pth'))
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
        try:
            model2.load_state_dict(torch.load(load_path + '/best.pth'))
            print("Restored weights from pretrained model2.")
        except Exception as e:
            print(e)
            print("Error occurred when restoring weights for model2.")
            val = ""
            while val != "y":
                val = input("Do you want to start training from scratch instead? (y/n):    ")
                if val == "n":
                    print("Aborting...")
                    exit()
    train(config, model, model2)


if __name__ == "__main__":
    main()
