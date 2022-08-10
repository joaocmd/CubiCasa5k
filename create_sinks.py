from floortrans.loaders import FloorplanSVG
from floortrans.loaders import svg_utils
from torch.utils.data import DataLoader
from xml.dom import minidom
from tqdm import tqdm
import numpy as np
import os
import pathlib
import cv2
import argparse
import pandas as pd

def distance(a, b):
    return np.linalg.norm(a-b)

def normalized(v):
    return v/np.linalg.norm(v)

def make_line_segment(XX, YY):
    p1, p2, p3, p4 = tuple(np.array([XX[i], YY[i]]) for i in range(len(XX)))

    if distance(p1, p2) < distance(p1, p3) and distance(p1, p2) < distance(p1, p4):
        s = (p1 + p2)/2
        e = (p3 + p4)/2
    elif distance(p1, p3) < distance(p1, p2) and distance(p1, p3) < distance(p1, p4):
        s = (p1 + p3)/2
        e = (p2 + p4)/2
    else:
        s = (p1 + p4)/2
        e = (p2 + p3)/2

    return sorted([s, e], key=lambda p: (p[0], -p[1]))

def crop_sink(img, el):
    _, _, X, Y = svg_utils.get_icon(el)
    pp = np.dstack((X, Y))[0]

    t = (pp[2] + pp[3])/2

    width_vector = pp[0]-pp[1]
    height_vector = pp[2]-pp[1]
    height = np.linalg.norm(height_vector)
    padding = 20

    c = np.mean(pp, axis=0)
    half_height = height_vector/2
    p0 = c - normalized(width_vector)*(height/2) - half_height
    p1 = c + normalized(width_vector)*(height/2) - half_height

    dst = np.array([[padding, padding+height], [padding+height, padding+height], [height/2+padding, padding]]) # t is on unit vector
    M = cv2.getAffineTransform(np.float32(np.array([p0, p1, t])), np.float32(dst))

    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=[1, 1, 1])
    cropped = cv2.getRectSubPix(rotated, np.int0(np.array([2*padding + height, 2*padding + height])), (height/2+padding, height/2+padding))
    return cropped

def print_stats(data_folder):
    data = pd.DataFrame(columns=['split', 'filename', 'idx', 'height', 'width'])
    data['height'] = data['height'].astype(int)
    data['width'] = data['width'].astype(int)

    examples = {}

    for split in os.listdir(data_folder):
        examples[split] = {}
        for filename in tqdm(os.listdir(data_folder + '/' + split)):
            if filename == 'A_wrong': # folder containing wrong examples
                continue
            
            idx = filename.split('_')[-1]
            sink_type = filename.split('_')[-2]
            
            if sink_type not in examples[split]:
                examples[split][sink_type] = 0
            examples[split][sink_type] += 1

            img = cv2.imread(f'{data_folder}/{split}/{filename}')
            height, width, _ = img.shape
            entry = [[split, filename, idx, height, width]]
            data = pd.concat([data, pd.DataFrame(entry, columns=data.columns)], ignore_index=True)

    print(data.describe(include='all'))

    print()
    print('split,width (mean),width (median),height (mean), height (median)')
    for split in os.listdir(data_folder):
        d = data[data['split'] == split]
        print(f"{split},{d['width'].mean()},{d['width'].median()},{d['height'].mean()},{d['height'].median()}")

    print(f"total,{data['width'].mean()},{data['width'].median()},{data['height'].mean()},{data['height'].median()}")

    print('\nsplit,type,count')
    for split in examples:
        for sink_type in sorted(examples[split]):
            print(f"{split},{sink_type},{examples[split][sink_type]}")

def main(data_folder, split, output_folder):
    output_folder = output_folder + '/' + split
    if not os.path.isdir(output_folder):
        pathlib.Path(output_folder).mkdir(parents=True)
    
    normal_set = FloorplanSVG(data_folder, split + '.txt', format='txt', original_size=False)
    data_loader = DataLoader(normal_set, batch_size=1, num_workers=0)
    data_iter = iter(data_loader)

    for val in tqdm(data_iter):
        image = val['image']
        image = np.moveaxis(image[0].numpy(), 0, -1) / 2 + 0.5
        folder = val['folder'][0]
        svg = minidom.parse(data_folder + folder + 'model.svg')
        sinks = [e for e in svg.getElementsByTagName("g")
                   if 'sink' in e.getAttribute("class").lower()]# and 'Corner' not in e.getAttribute("class")]

        for idx, e in enumerate(sinks):
            cropped = crop_sink(image, e)

            cv2.imwrite(f'{output_folder}/{folder[1:-1].replace("/","_")}_{e.getAttribute("class").split(" ")[1]}_{idx}.png', (cropped*255).astype(np.uint8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-folder', nargs='?', type=str, default='data/cubicasa5k/')
    parser.add_argument('--split', nargs='?', type=str, default='all')
    parser.add_argument('--out-folder', nargs='?', type=str, default='sinks/')
    parser.add_argument('--stats', nargs='?', type=bool, default=False, const=True)
    parser.add_argument('--yes', nargs='?', type=bool, default=False, const=True)

    args = parser.parse_args()

    if args.stats:
        print_stats(args.out_folder)
    else:
        if args.yes or not input(f'Resume? This will overwrite current {args.out_folder}{args.split}').lower().startswith('n'):
            if args.split == 'all':
                for split in ('train', 'val', 'test'):
                    print(f'Starting split: {split}')
                    main(args.data_folder, split, args.out_folder)