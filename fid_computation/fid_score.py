import os
import click
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import scipy
from tqdm import tqdm
from inception import InceptionV3


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, res=256):
        self.root_dir = root_dir
        self.image_paths = self._load_image_paths(root_dir)
        self.res = res

    def _load_image_paths(self, root_dir):
        image_paths = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.npy')

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(dirpath, filename))
        
        return image_paths
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        if img_path[-4:] == '.npy':
            image = torch.tensor(np.load(img_path))
            return image
        else:
            image = Image.open(img_path).convert('RGB').resize((self.res, self.res))

            return transforms.functional.to_tensor(image)


@torch.no_grad()
def compute_statistics(path, model, batch_size=50, dims=2048, device='cuda', num_workers=8, save_path=''):
    """
    Heavily inspired from https://github.com/NVlabs/edm
    """
    model.eval()
    dataset = ImageFolderDataset(path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    N = 0
    mu = 0
    sigma = 0
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        features = model(batch)[0].to(torch.float64).reshape(batch.shape[0], dims)
        mu += features.sum(0)
        sigma += features.T @ features
        N+= batch.shape[0]


    mu /= N
    sigma -= mu.ger(mu) * N
    sigma /= (N - 1)

    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()
    np.save(os.path.join(save_path,'fid_stats'), {'mu' : mu, 'sigma' : sigma})
    return  mu, sigma



def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))



@click.command()
@click.option('--path', type=str)
@click.option('--ref_path', type=str)
@click.option('--save_path', type=str)
@click.option('--has_stats', type=bool, default=False)
@click.option('--has_ref_stats', type=bool, default=True)
@click.option('--batch_size', type=int, default=50)
@click.option('--num_workers', type=int, default=4)
def calculate_fid_given_paths(path, ref_path, has_stats, has_ref_stats, batch_size, num_workers, save_path):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)
    os.makedirs(os.path.join(save_path, 'ref'),exist_ok=True)
    if has_stats:
        fid_stats = np.load(path, allow_pickle=True).item()
        m2 = fid_stats['mu']
        s2 = fid_stats['sigma']
    else:
        m1, s1 = compute_statistics(path, model, batch_size,
                                        dims, device, num_workers, save_path)
    if has_ref_stats:
        fid_stats = np.load(ref_path, allow_pickle=True).item()
        m2 = fid_stats['mu']
        s2 = fid_stats['sigma']
    else:
        m2, s2 = compute_statistics(ref_path, model, batch_size,
                                        dims, device, num_workers, os.path.join(save_path,'ref'))
    fid_value = calculate_fid_from_inception_stats(m1, s1, m2, s2)
    print(fid_value)
    return fid_value


if __name__ == '__main__':
    calculate_fid_given_paths()