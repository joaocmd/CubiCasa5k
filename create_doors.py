from floortrans.loaders import FloorplanSVG
from torch.utils.data import DataLoader
from xml.dom import minidom
from tqdm import tqdm
import numpy as np
import os
import cv2

from PIL import Image

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
    orientation = "p" if np.cross(v, normal)[2] < 0 else "n" 

    return axis + orientation

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
            return 'Double-' + classify_panel(s, e, panels[0])[1] # doublep | doublen
        else:
            return 'Single-' + classify_panel(s, e, panels[0]) # lp | ln | rp | rn
    if cls == 'Door Swing Opposite':
        return 'Opposite-' + classify_panel(s, e, panels[0]) + classify_panel(s, e, panels[1])

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
    v = np.hstack((e - s, [0]))

    n = np.array([-v[1], v[0]]) # counterclockwise perpendicular
    n /= np.linalg.norm(n)
    t = centroid + n

    # print(s, e, v , n, t)
    width = np.linalg.norm(v)
    padding = 20
    dst = np.array([[padding, padding+width], [padding+width, padding+width], [width/2+padding, padding+width+1]]) # t is on unit vector
    M = cv2.getAffineTransform(np.float32(np.array([s, e, t])), np.float32(dst))

    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=[1, 1, 1])
    cropped = cv2.getRectSubPix(rotated, np.int0(np.array([width+2*padding, 2*(width+padding)])), (width/2+padding, padding+width))
    return cropped

def main(data_folder, data_file, output_folder):

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    normal_set = FloorplanSVG(data_folder, data_file, format='txt', original_size=False)
    data_loader = DataLoader(normal_set, batch_size=1, num_workers=0)
    data_iter = iter(data_loader)

    for val in tqdm(data_iter):
        image = val['image']
        image = np.moveaxis(image[0].numpy(), 0, -1) / 2 + 0.5
        folder = val['folder'][0]
        svg = minidom.parse(data_folder + folder + 'model.svg')
        doors = [e for e in svg.getElementsByTagName("g") if e.getAttribute("id") == "Door"]

        for door_idx, d in enumerate(doors):
            cls = d.getAttribute("class")
            cropped = crop_door(image, d)
            classification = classify_door(d)

            im = Image.fromarray((cropped*255).astype(np.uint8))
            im.save(f'{output_folder}{folder[1:-1].replace("/","_")}_{door_idx}-{classification}.png')

if __name__ == '__main__':
    data_folder = 'data/cubicasa5k/'
    data_file = 'train.txt'
    output_folder = 'doors/'
    main(data_folder, data_file, output_folder)


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