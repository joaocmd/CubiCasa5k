import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from floortrans.loaders.svg_loader import FloorplanSVG
from torch.utils import data

def main(args):
    dataset = FloorplanSVG(args.data_path, args.txt, format='lmdb', original_size=True, is_transform=False, return_heatmaps=True)
    loader = data.DataLoader(dataset, shuffle=False, num_workers=0, batch_size=1)

    total_size = 0
    means = [0, 0, 0]
    for elem in tqdm(loader):
        img = elem['image'].squeeze(0)
        size = img.shape[1]*img.shape[2]
        for i in range(3):
            means[i] += img[i].sum(dtype=float).item()
        total_size += size

    for i in range(3):
        means[i] /= total_size
    # means = [235.9335, 235.9548, 234.9813]

    total_size = 0
    stds = [0, 0, 0]
    for elem in tqdm(loader):
        img = elem['image'].squeeze(0)
        size = img.shape[1]*img.shape[2]
        for i in range(3):
            stds[i] += ((img[i] - means[i])**2).sum().item()
        total_size += size

    for i in range(3):
        stds[i] = np.sqrt(stds[i]/total_size)
    # stds = [52.206464353899975, 51.46838985233353, 52.863078274595516]
    
    print(means)
    print(stds)


if __name__ == '__main__':
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description='Script for getting pixel mean and variance.')
    parser.add_argument('--txt', nargs='?', type=str, default='train.txt',
                        help='Path to text file containing file paths')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='Path to data directory')
    args = parser.parse_args()

    main(args)
