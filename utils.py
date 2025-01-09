import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F

import numpy as np

# Visuals utils
import os
import matplotlib.pyplot as plt




# 6. Visualize a Few Test Images and Their Predictions
def visualize_predictions(images):

    # Plot the images and predicted labels
    fig = plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = fig.add_subplot(3, 3, i+1)
        ax.imshow(images[i].numpy().squeeze(), cmap='gray')
        ax.axis('off')
    
    plt.show()


def visualize_errors(true_seg, pred_seg, title):
    # batch_size = batch.shape[0]
    samples = 8

    fig, axes = plt.subplots(samples, 2, figsize=(10, 20))  # Adjust figsize to accommodate more rows
    fig.suptitle(title, fontsize=16)


    for i in range(samples):
        axes[i,0].imshow(true_seg[i].squeeze(), cmap = 'gray')
        axes[i,0].axis('off')

        axes[i,1].imshow(pred_seg[i].squeeze(),cmap = 'gray')
        axes[i,1].axis('off')


    row_titles = ['Ground truth', 'Vq-Vae predictions']
    for i in range(2):
        axes[0, i].set_title(row_titles[i], fontsize=14, fontweight='bold')
    
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def evaluate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (images, classes) in val_loader:
            inputs = images.float().to(device)
           
            outputs, _, _, _ = model(inputs)
            
            # Loss and backward
            loss = F.mse_loss(inputs, outputs)
            
            val_loss += loss.item()

    epoch_val_loss = val_loss / len(TestLoader.dataset)

    return epoch_val_loss



def plot_train_val_loss(train_loss_values, val_loss_values ):
    # Plot the training and validation losses
    plt.figure(figsize=(20, 10))
    plt.plot(train_loss_values, label='Train Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Evolution of Loss')
    plt.legend()
    plt.grid()
    plt.show()



def save_model(model_name, model, epoch, train_loss_values, val_loss_values, codebook_loss_values):
    checkpoint_path = os.path.join( os.getcwd() , model_name )
    torch.save({'epoch' : epoch,
                'K' : model.vq_layer.K,
                'D' :  model.vq_layer.D,
                'model_state_dict' : model.state_dict(),
                'train_loss_values' : train_loss_values, 
                'val_loss_values' : val_loss_values, 
                'codebook_loss_values' : codebook_loss_values,
                'codebook' : model.vq_layer.embedding.weight.data }, checkpoint_path)




def codebook_hist_testset(model, val_loader):
    model.eval()
    hist = torch.zeros(model.vq_layer.K).to(device)

    with torch.no_grad():
        for (batch, classes) in val_loader:
            hist += model.codebook_usage(batch.float().to(device))
    
    hist = hist.detach().cpu().numpy()
    unused_codes = len(np.where(hist == 0.0)[0])

    percentage = (model.vq_layer.K - unused_codes)*100/model.vq_layer.K

    print(f" ONLY {model.vq_layer.K - unused_codes} OF CODES WERE USED FROM {model.vq_layer.K}, WHICH MAKE {percentage} % OF CODES FROM THE CODE-BOOK")
    return hist




def evaluate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (images, classes) in val_loader:
            inputs = images.float().to(device)
           
            outputs, _, _, _ = model(inputs)
            
            # Loss and backward
            loss = F.mse_loss(inputs, outputs)
            
            val_loss += loss.item()

    epoch_val_loss = val_loss / len(TestLoader.dataset)

    return epoch_val_loss