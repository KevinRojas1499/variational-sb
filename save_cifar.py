import os
import numpy as np
import torch
import click
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.autoencoder import Autoencoder
import PIL.Image

def get_dataset(name, res):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(res)])
    if name == 'cifar':
        train_dataset = datasets.CIFAR10(root='./saved_datasets', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./saved_datasets', train=False, download=True, transform=transform)
        return train_dataset, test_dataset, 3
    elif name == 'mnist':
        train_dataset = datasets.MNIST(root='./saved_datasets', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./saved_datasets', train=False, download=True, transform=transform)
        return train_dataset, test_dataset, 1

@click.command()
@click.option('--dataset', type=click.Choice(['cifar','mnist']), default='cifar')
@click.option('--starting_res', type=int, default=32)
@click.option('--batch_size', type=int, default=256)
def train(dataset, starting_res,  batch_size ):
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset, in_channels = get_dataset(dataset, starting_res) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=False, drop_last=True)

    autoencoder = Autoencoder('cifar_vae_big_long').to(device)

    autoencoder.eval()
    pbar = tqdm(train_loader, total=len(train_dataset)//batch_size )
    cur_batch = 0
    for images, _ in pbar:
        images = images.to(device)
    
        # decoded = images
        encoded = autoencoder.encode(images)
        decoded = autoencoder.decode(encoded)
        
        images_np = ( decoded * 255 ).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for i in range(batch_size):
            folder = f'cifar_auto_np/{cur_batch}/'
            os.makedirs(folder, exist_ok=True)
            
            # np.save(os.path.join(folder, f'{i}.npy'), images_np[i]) 
            PIL.Image.fromarray(images_np[i], 'RGB').save(os.path.join(folder,f'{i}.png'))

        cur_batch+=1

if __name__ == '__main__':
    train()
