import lmdb
import os
import pickle
import argparse
import logging
import numpy as np
from datetime import datetime
from tqdm import tqdm
from floortrans.loaders.svg_loader import FloorplanSVG
import torch

MAX_ROOMS = 11
MAX_ICONS = 10
MAX_CLASSES = {
    0: MAX_ROOMS,
    1: MAX_ICONS
}

def verify_example(txt_sample, lmdb_sample, idx=0):
    assert idx in (0, 1) # 0 is rooms, 1 is icons
    MAX = MAX_CLASSES[idx]

    txt_label = txt_sample["label"].data.numpy()[idx]
    lmdb_label = lmdb_sample["label"].data.numpy()[idx]

    lm_labels_mask = np.where(lmdb_label > MAX)
    updated = np.any(lm_labels_mask)
    if updated:
        # Correct labels with the ones from the original files
        new_labels = lmdb_label.copy()
        new_labels[lm_labels_mask] = txt_label[lm_labels_mask]

        if (new_labels != txt_label).any():
            print("Still not good!")
            raise ValueError()

        return updated, new_labels
    return updated, None

# TODO: Add verify the labels are the same!

def main(args, logger):
    logger.info("Opening database...")
    env = lmdb.open(args.lmdb)

    logger.info("Creating data loader...")
    train_data = FloorplanSVG(args.data_path, f"{args.split}.txt", format='txt', original_size=True, is_transform=False)
    filepaths = train_data.folders

    update_counts = 0
    new_counts = 0
    for index, filepath in tqdm(enumerate(filepaths), total=len(filepaths)):
        key = filepath.encode("ascii")
        txt_sample = train_data[index]
        assert filepath == txt_sample["folder"], "filepath mismatch"

        with env.begin(write=False) as f:
            data = f.get(key)

        if data is not None: # Update
            lmdb_sample = pickle.loads(data)
            to_update_rooms, update_labels_rooms = verify_example(txt_sample, lmdb_sample, idx=0)
            if to_update_rooms:
                    logger.info(f'--> Update rooms for {filepath}')
                    #lmdb_sample["label"][0] = torch.from_numpy(update_labels_rooms).to(lmdb_sample["label"][0])
        
            to_update_icons, update_labels_icons = verify_example(txt_sample, lmdb_sample, idx=1)
            if to_update_icons:
                    logger.info(f'--> Update rooms for {filepath}')
                    #lmdb_sample["label"][1] = torch.from_numpy(update_labels_icons).to(lmdb_sample["label"][1])
                    
            if to_update_rooms or to_update_icons:
                logger.info(f'--> Update labels: {lmdb_sample["label"].shape}')
                update_counts += 1
        else:
            logger.info(f"Missing from database {filepath}")
            new_counts += 1

    logger.info(f"Total new: {new_counts}, updates: {update_counts}")


if __name__ == '__main__':
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    parser = argparse.ArgumentParser(description='Script for creating lmdb database.')
    parser.add_argument('--lmdb', nargs='?', type=str,
                        default='data/cubicasa5k/cubi_lmdb/', help='Path to lmdb')
    parser.add_argument('--data-path', nargs='?', type=str, default='data/cubicasa5k/',
                        help='Path to data directory')
    parser.add_argument('--split', nargs='?', type=str, required=True,
                    help='Split to run the verifier for')
    parser.add_argument('--log-path', nargs='?', type=str, default='runs_cubi/',
                        help='Path to log directory')
    args = parser.parse_args()

    log_dir = args.log_path + '/' + time_stamp + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger('lmdb')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_dir+'/lmdb.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    main(args, logger)
