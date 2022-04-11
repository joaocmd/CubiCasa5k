from floortrans.loaders import FloorplanSVG
from torch.utils.data import DataLoader
from xml.dom import minidom
from tqdm import tqdm
import numpy as np
import os
import pathlib
import cv2
import argparse

def distance(a, b):
    return np.linalg.norm(a-b)

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

def classify_panel(s, e, panel):
    centroid = (s + e)/2
    v = np.hstack((e - s, [0]))

    path = next(p for p in panel.childNodes if p.nodeName == "path").getAttribute("d")

    startpoint = np.array(eval(path.split(' ')[0][1:]))
    endpoint = np.array(eval(path.split(' ')[2]))
    endpoint = startpoint + endpoint

    axis = "l" if distance(s, endpoint) < distance(e, endpoint) else "r"

    normal = np.hstack((endpoint - centroid, [0]))
    orientation = "u" if np.cross(v, normal)[2] < 0 else "d"

    return orientation + axis

def classify_door(el, panel_idx=0):
    cls = el.getAttribute('class')
    if cls not in ('Door Swing Beside', 'Door Swing Opposite'):
        return cls

    pol = next(p for p in el.childNodes if p.nodeName == 'polygon')
    points = pol.getAttribute('points').split(' ')
    points = points[:-1]

    X, Y = np.array([]), np.array([])
    for a in points:
        x, y = a.split(',')
        X = np.append(X, float(x))
        Y = np.append(Y, float(y))

    s, e = make_line_segment(X, Y)

    panels = [p for p in el.childNodes if p.getAttribute("id") == "Panel"]
    if cls == 'Door Swing Beside':
        if len(panels) > 1:
            return 'Double_' + classify_panel(s, e, panels[0])[0] # doubleu | doubled
        else:
            return 'Single_' + classify_panel(s, e, panels[0]) # ul | ur | dl | dr
    if cls == 'Door Swing Opposite':
        return 'Opposite_' + '_'.join(sorted((classify_panel(s, e, panels[0]), classify_panel(s, e, panels[1])), reverse=True))

def crop_door(img, el):
    pol = next(p for p in el.childNodes if p.nodeName == "polygon")
    points = pol.getAttribute("points").split(' ')
    points = points[:-1]

    X, Y = np.array([]), np.array([])
    for a in points:
        x, y = a.split(',')
        X = np.append(X, float(x))
        Y = np.append(Y, float(y))

    s, e = make_line_segment(X, Y)
    centroid = (s + e)/2
    v = e - s

    n = np.array([-v[1], v[0]]) # counterclockwise perpendicular
    n /= np.linalg.norm(n)
    t = centroid + n

    # print(s, e, v , n, t)
    width = np.linalg.norm(s-e)
    padding = 20
    dst = np.array([[padding, padding+width], [padding+width, padding+width], [width/2+padding, padding+width+1]]) # t is on unit vector
    M = cv2.getAffineTransform(np.float32(np.array([s, e, t])), np.float32(dst))

    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=[1, 1, 1])
    cropped = cv2.getRectSubPix(rotated, np.int0(np.array([width+2*padding, 2*(width+padding)])), (width/2+padding, padding+width))
    return cropped

def stats(data_folder):
    data = {}
    for split in os.listdir(data_folder):
        split_data = {}
        for filename in tqdm(os.listdir(data_folder + '/' + split)):
            cls = filename.split('-')[1].split('.')[0]
            if cls not in split_data:
                split_data[cls] = 0
            split_data[cls] += 1
        data[split] = split_data

    return data

def rename(cls, flip):
    replacements = {
        'vertical': str.maketrans('udlr', 'dulr'),
        'horizontal': str.maketrans('udlr', 'udrl'),
        'both': str.maketrans('udlr', 'durl'),
    }

    if 'Double' in cls:
        name, orientation = cls[:-1], cls[-1:]
    elif 'Opposite' in cls:
        name, orientation = cls[:-5], cls[-5:]
    elif 'Single' in cls:
        name, orientation = cls[:-2], cls[-2:]
    else:
        name, orientation = cls, ''

    orientation = orientation.translate(replacements[flip])

    return name + '_'.join(sorted(orientation.split('_'), reverse=True))


def augment(data_folder):
    for filename in tqdm(os.listdir(data_folder)):
        name = filename.split('-')[0]
        cls = filename.split('-')[1].split('.')[0]

        if 'Single' in cls:
            continue

        img = cv2.imread(data_folder + filename)
        vertical = cv2.flip(img, 0)
        horizontal = cv2.flip(img, 1)
        both = cv2.flip(img, -1)

        cv2.imwrite(data_folder + f'/AUGvertical_{name}-{rename(cls, "vertical")}.png', vertical)
        cv2.imwrite(data_folder + f'/AUGhorizontal_{name}-{rename(cls, "horizontal")}.png', horizontal)
        cv2.imwrite(data_folder + f'/AUGboth_{name}-{rename(cls, "both")}.png', both)
            
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
        doors = [e for e in svg.getElementsByTagName("g") if e.getAttribute("id") == "Door"]

        for door_idx, d in enumerate(doors):
            cropped = crop_door(image, d)
            classification = classify_door(d)

            cv2.imwrite(f'{output_folder}/{folder[1:-1].replace("/","_")}_{door_idx}-{classification}.png', (cropped*255).astype(np.uint8))

def or_zero(x):
   return x if x else 0

def all_keys(stats):
    splits = stats.keys()
    all_classes = set()
    for s in splits:
        all_classes = all_classes.union(set(stats[s].keys()))
    return all_classes

def get_value(stats, split, cls):
    if cls in stats[split]:
        return stats[split][cls]
    return 0
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-folder', nargs='?', type=str, default='data/cubicasa5k/')
    parser.add_argument('--split', nargs='?', type=str, default='test')
    parser.add_argument('--out-folder', nargs='?', type=str, default='doors/')
    parser.add_argument('--stats', nargs='?', type=bool, default=False, const=True)
    parser.add_argument('--augment', nargs='?', type=bool, default=False, const=True)
    parser.add_argument('--yes', nargs='?', type=bool, default=False, const=True)

    args = parser.parse_args()

    if (args.stats):
        data_stats = stats(args.out_folder)

        print('class,train,val,test')
        for k in sorted(all_keys(data_stats)):
            print(f'{k},' + ','.join(map(str, (get_value(data_stats, 'train', k), get_value(data_stats, 'val', k), get_value(data_stats, 'test', k)))))
    elif (args.augment):
        augment(args.out_folder)
    else:
        if args.yes or not input(f'Resume? This will overwrite current {args.out_folder}{args.split}').lower().startswith('n'):
            main(args.data_folder, args.split, args.out_folder)
    # print(data_stats)


    # files = np.genfromtxt(data_file, dtype='str')
    # classes = set()
    # for file in tqdm(files):
    #     svg = minidom.parse(data_folder+file+'model.svg')
    #     classes.update(e.getAttribute('class') for e in svg.getElementsByTagName('g'))
    # door_classes = [
    #     'Door Zfold Beside', # folding doors
    #     'Door None Beside', # weird openings
    #     'Door Swing Opposite', # opens in both directions opposite
    #     'Door Swing Beside', # normal swing door, opens in one direction
    #     'Door ParallelSlide Beside', # parallel sliding door
    #     'Door Slide Beside', # sliding door
    #     'Door RollUp Beside' # rollup doors
    # ]