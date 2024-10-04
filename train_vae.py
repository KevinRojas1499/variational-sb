import torch
import click
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP 

def plot_samples(images, encoded, reconstructed_images):
    for j in range(32):
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(encoded[j].clamp(0,1).permute(1,2,0).cpu().detach().numpy())
        ax[1].imshow(images[j].clamp(0,1).permute(1,2,0).cpu().detach().numpy())
        ax[2].imshow(reconstructed_images[j].clamp(0,1).permute(1,2,0).cpu().detach().numpy())
                        
        fig.savefig(f'samples/sample_{j}.png')
        plt.close(fig)
        
def loss_fn(autoencoder, images, al, return_images=False):
    reconstruction_loss_fn = nn.MSELoss()
    latent_dist = autoencoder.module.encode(images).latent_dist
    encoded = latent_dist.sample()
    reconstructed_images = autoencoder.module.decode(encoded).sample
    # Compute losses
    reconstruction_loss = reconstruction_loss_fn(reconstructed_images, images)
    kl_loss = 0.5 * torch.mean( latent_dist.mean.pow(2) + latent_dist.logvar.exp() - latent_dist.logvar - 1)
    loss =  al * reconstruction_loss + (1-al)* kl_loss
    if return_images:
        return loss, encoded, reconstructed_images
    else:
        return loss

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
@click.option('--num_blocks', type=int, default=2)
@click.option('--latent_channels', type=int, default=4)
@click.option('--batch_size', type=int, default=256)
@click.option('--num_epochs', type=int, default=50)
@click.option('--al', type=float, default=.99999)
@click.option('--seed', type=int, default=42)
@click.option('--dir', type=str, default=None)
def train(dataset, starting_res, latent_channels, num_blocks, batch_size, num_epochs, al, seed, dir):
    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    assert batch_size % world_size == 0, f"Batch size must be divisible by world size."
    batch_size//=world_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    if dir is None:
        dir = f'./{dataset}_vae'
        
    train_dataset, test_dataset, in_channels = get_dataset(dataset, starting_res) 
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler = test_sampler, shuffle=False, drop_last=True)

    down_blocks = ('DownEncoderBlock2D',) * num_blocks
    up_blocks = ('UpDecoderBlock2D',) * num_blocks
    out_channels = [ starting_res//2**i for i in range(num_blocks)]
    autoencoder = AutoencoderKL(in_channels= in_channels, out_channels=in_channels,
                                latent_channels=latent_channels,
                                down_block_types=down_blocks,block_out_channels=out_channels,
                                layers_per_block=2,
                                up_block_types=up_blocks, norm_num_groups=out_channels[-1]).to(device=device)
    autoencoder = DDP(autoencoder) 
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        autoencoder.train()
        total_loss = 0
        pbar = tqdm(train_loader, total=len(train_dataset)//(batch_size * world_size)) if rank == 0 else train_loader
        for images, _ in pbar:
            images = images.to(device)  # Move images to GPU if available
            loss = loss_fn(autoencoder, images, al)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            dist.all_reduce(loss) 
            loss/= world_size
            total_loss += loss.item()
            if rank == 0:
                pbar.set_description(f'Loss : {loss.item() : .5f}')
        
        autoencoder.module.save_pretrained(dir)
        
        # Eval section
        if epoch%10 == 0 or epoch == num_epochs - 1:
            autoencoder.eval()
            with torch.no_grad():
                total_loss = 0
                k = 0

                for images, _ in test_loader:
                    images = images.to(device)
                    
                    loss, encoded, reconstructed_images = loss_fn(autoencoder, images, al, return_images=True)

                    dist.all_reduce(loss)
                    loss = loss/world_size
                    total_loss += loss.item()
                    
                    if k == 0:
                        k+=1
                        plot_samples(images, encoded, reconstructed_images)
                
                print(f"Test Loss: {total_loss/len(test_loader):.5f}")
        dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    train()
