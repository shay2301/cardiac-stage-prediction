import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

def convert_to_list(input_string):
    matches = re.findall(r'\[(\d+) (\d+)\]', input_string)
    if len(matches) == 0:
        matches = re.findall(r'\[( \d+) ( \d+)\]', input_string)
    number_list = [list(map(int, match)) for match in matches]
    return number_list

def imshow(batch, esv_edv_pred_imgs=None, alpha=0.5):
    esv_imgs = batch['frames']['esv']
    edv_imgs = batch['frames']['edv']
    esv_masks = batch['masks']['esv']
    edv_masks = batch['masks']['edv']
    esv_imgs = [np.transpose(img.numpy(), (1, 2, 0)) for img in esv_imgs]
    edv_imgs = [np.transpose(img.numpy(), (1, 2, 0)) for img in edv_imgs]
    esv_masks = [mask.numpy() for mask in esv_masks]
    edv_masks = [mask.numpy() for mask in edv_masks]
    esv_rgba_masks = [np.zeros((mask.shape[1], mask.shape[2], 4), dtype=np.float32) for mask in esv_masks]
    edv_rgba_masks = [np.zeros((mask.shape[1], mask.shape[2], 4), dtype=np.float32) for mask in edv_masks]
    for i, (edv_mask, esv_mask) in enumerate(zip(edv_masks, esv_masks)):
        esv_rgba_masks[i][esv_mask.squeeze() > 0] = [0, 0, 1, alpha]
        edv_rgba_masks[i][edv_mask.squeeze() > 0] = [0, 0, 1, alpha]

        esv_rgba_masks[i][esv_mask.squeeze() == 0] = [0, 0, 0, 0]
        edv_rgba_masks[i][edv_mask.squeeze() == 0] = [0, 0, 0, 0]
    if esv_edv_pred_imgs:
        esv_pred_imgs = esv_edv_pred_imgs[0]
        edv_pred_imgs = esv_edv_pred_imgs[1]
        esv_pred_rgba_masks = [np.zeros((mask.shape[1], mask.shape[2], 4), dtype=np.float32) for mask in esv_masks]
        edv_pred_rgba_masks = [np.zeros((mask.shape[1], mask.shape[2], 4), dtype=np.float32) for mask in edv_masks]
        for i, (esv_pred_img, edv_pred_img) in enumerate(zip(esv_pred_imgs, edv_pred_imgs)):
            esv_pred_rgba_masks[i][esv_pred_img.squeeze() > 0.5] = [0, 1, 0, alpha]
            edv_pred_rgba_masks[i][edv_pred_img.squeeze() > 0.5] = [0, 1, 0, alpha]

            esv_pred_rgba_masks[i][esv_pred_img.squeeze() <= 0.5] = [0, 0, 0, 0]
            edv_pred_rgba_masks[i][edv_pred_img.squeeze() <= 0.5] = [0, 0, 0, 0]

    fig, axes = plt.subplots(2, len(esv_imgs), figsize=(20, 20))
    plt.subplots_adjust(wspace=0, hspace=-0.5)
    for i, (esv_img, edv_img, esv_mask, edv_mask, filename) in enumerate(zip(esv_imgs, edv_imgs, esv_rgba_masks, edv_rgba_masks, batch['FileName'])):
        axes[0, i].imshow(esv_img, cmap='gray')
        axes[0, i].imshow(esv_mask)
        axes[0, i].imshow(esv_pred_rgba_masks[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(filename)
        axes[1, i].imshow(edv_img, cmap='gray')
        axes[1, i].imshow(edv_mask)
        axes[1, i].imshow(edv_pred_rgba_masks[i])
        axes[1, i].axis('off')
    # axes[0, 0].set_ylabel('ESV', labelpad=15)
    # axes[1, 0].set_ylabel('EDV', labelpad=15)
    fig.text(0.04, 0.5, 'ESV', va='center', rotation='vertical')  # Set the row title for ESV
    fig.text(0.04, 0.25, 'EDV', va='center', rotation='vertical')  # Set the
    plt.show()



class Patient():
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.video_path = None
        self.fps = None
        self.esv_idx = None
        self.edv_idx = None
        self.esv_poly = None
        self.edv_poly = None
        self.esv_frame = None
        self.edv_frame = None
        self.esv_mask = None
        self.edv_mask = None
        self.split = None

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, input_transform=None, target_transform=None,split_group=None, split_method=None, test_size=0.2, val_size=0.2):
        self.root_dir = root_dir
        csv_file = os.path.join(root_dir, 'Database', 'training_database.csv')
        self.df = pd.read_csv(csv_file)
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.split_group = split_group
        self.split_method = split_method
        self.test_size = test_size
        self.val_size = val_size

        if not self.split_method:
            self.df_train = self.df[self.df['Split'].str.lower() == 'train']
            self.df_val = self.df[self.df['Split'].str.lower() == 'val']
            self.df_test = self.df[self.df['Split'].str.lower() == 'test']
        elif self.split_method=='EF':
            self.df['EF_bins'] = pd.cut(self.df['EF'], bins=10)
            self.df_train, self.df_test = train_test_split(self.df, test_size=self.test_size, stratify=self.df['EF_bins'], random_state=42)
            self.df_train, self.df_val = train_test_split(self.df_train, test_size=self.val_size/(1-self.test_size), stratify=self.df_train['EF_bins'], random_state=42)
        
        if self.split_group == 'train':
            self.df = self.df_train
        elif self.split_group == 'val':
            self.df = self.df_val
        elif self.split_group == 'test':
            self.df = self.df_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient = Patient(row['FileName'].replace('.avi', ''))
        patient.video_path = os.path.join(self.root_dir, 'EchoNet-Dynamic', 'Videos', row['FileName'])
        patient.esv_idx = row['ESV_frame']
        patient.edv_idx = row['EDV_frame']
        patient.esv_poly = row['ESV_polygon']
        patient.edv_poly = row['EDV_polygon']


        cap = cv2.VideoCapture(patient.video_path)
        
        for index in [patient.esv_idx, patient.edv_idx]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if ret:
                if self.input_transform:
                    if patient.esv_idx == index:
                        patient.esv_frame = self.input_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    elif patient.edv_idx == index:
                        patient.edv_frame = self.input_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                else:
                    if patient.esv_idx == index:
                        patient.esv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    elif patient.edv_idx == index:
                        patient.edv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cap.release()

        channels, height, width = patient.esv_frame.shape


        patient.esv_mask = np.zeros((height, width), dtype=np.uint8)
        patient.edv_mask = np.zeros((height, width), dtype=np.uint8)

        
        cv2.fillPoly(patient.esv_mask, [np.array(convert_to_list(patient.esv_poly))], 255)
        cv2.fillPoly(patient.edv_mask, [np.array(convert_to_list(patient.edv_poly))], 255)

        patient.esv_mask = np.expand_dims(patient.esv_mask, axis=2)
        patient.edv_mask = np.expand_dims(patient.edv_mask, axis=2)

        if self.target_transform:
            patient.esv_mask = self.target_transform(patient.esv_mask)
            patient.edv_mask = self.target_transform(patient.edv_mask)

        return {'FileName': patient.patient_id,
            'frames': {'esv': patient.esv_frame,
                   'edv': patient.edv_frame},
        'masks': {'esv': patient.esv_mask,
                  'edv': patient.edv_mask}}

class FullVideoDataset(Dataset):
    def __init__(self, root_dir, input_transform=None):
        self.root_dir = root_dir
        csv_file = os.path.join(root_dir, 'Database', 'training_database.csv')
        self.df = pd.read_csv(csv_file)
        self.input_transform = input_transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient = Patient(row['FileName'].replace('.avi', ''))
        patient.video_path = os.path.join(self.root_dir, 'EchoNet-Dynamic', 'Videos', row['FileName'])
        patient.fps = row['FPS']
        
        cap = cv2.VideoCapture(patient.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for frame_idx in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                if self.input_transform:
                    frames.append(self.input_transform(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
                else:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                raise Exception(f"Failed to read frame {frame_idx} from video {patient.video_path}")
        cap.release()

        # Pad frames if video is shorter than max_length
        max_length = 1002
        while len(frames) < max_length:
            frames.append(torch.zeros_like(frames[0]))

        return {'patient_id': patient.patient_id, 'frames': torch.stack(frames), 'fps': patient.fps}