"""
This file contains a function to plot sample images and predictions and a function ot visualize filters from the first conv layer

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from label_map import idx_to_clss
from torchvision.utils import make_grid


# ================================================================================================
# Function to plot sample images and predictions


def plot_sample_images(test_dataloader, model): 
    # test dataloader should have batch size as 200 (number of images per class)
    predictions = {}
    classwise_acc = {}
    pred_samples = {}
    prob_samples = {}
    sample_images = {}
    model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
            # Prediction on the test set batch
            x, y = batch
            logits = torch.nn.functional.softmax(model(x), dim=1)
            prob, preds = torch.max(logits, dim=1)

            # Randomly selecting three images from a class
            rand_sam = torch.randperm(x.size(0))[:3]    
            sample_images[y[0].item()] = x[rand_sam, :, :, :]      # Saving predictions, probabilitis and accuracies
            pred_samples[y[0].item()] = preds[rand_sam]
            prob_samples[y[0].item()] = prob[rand_sam]
            predictions[y[0].item()] = preds
            classwise_acc[y[0].item()] = torch.sum(preds == y)/preds.size(0)


    # Plotting samples from each class in 10 x 3 grid

    fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(3 * 3, 3 * 10))

    for i, (label, imgs) in enumerate(sample_images.items()):
        for j in range(3):
            ax = axes[i, j]
            img = imgs[j]
            pred = pred_samples[label][j]
            prob = prob_samples[label][j]

            ax.imshow(img.permute(1,2,0))
            ax.axis('off')
            ax.set_title(f'{idx_to_clss[pred.item()]} ({prob*100:.1f}%)', fontsize=10)

            # True label as row title
            if j == 0:
                ax.text(-0.05, 0.5, idx_to_clss[label], fontsize=12, ha='right', va='center',
                        transform=ax.transAxes, rotation=90, fontweight='bold')
            
            # Class wise accuracy 
            if j == 3 - 1:
                accuracy = classwise_acc[label]
                ax.text(1.2, 0.5, f"Class accuracy: {accuracy*100:.1f}%", fontsize=10,
                            transform=ax.transAxes, rotation=90, va='center', color='black')

    plt.tight_layout()
    wandb.log({"Class wise predictions": wandb.Image(fig)})





# ================================================================================
def visualize_filters(test_dataloader, model):  
    # Visualize filters in the first layer of the model
    for batch in test_dataloader:
        with torch.no_grad():
            x, _ = batch
            random_img = np.random.randint(0, x.shape[0])
            filters = model.model.convblock1(x[random_img : random_img+1, :, :,:])
            break


    img_grid = make_grid(filters.permute(1,0,2,3), nrow=4, padding=4).permute(1,2,0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    ax.imshow(img_grid)
    ax.axis('off')
    plt.tight_layout()
    wandb.log({"Layer 1 filters": wandb.Image(fig)})

 # ======================================================================================