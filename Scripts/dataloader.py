import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# def imshow(img, mask=None, FileName=None, is_esv=True):
def imshow(batch, alpha=0.5):
    esv_imgs = batch['frames']['esv']
    edv_imgs = batch['frames']['edv']
    esv_masks = batch['masks']['esv']
    edv_masks = batch['masks']['edv']
    esv_imgs = [np.transpose(img.numpy(), (1, 2, 0)) for img in esv_imgs]
    edv_imgs = [np.transpose(img.numpy(), (1, 2, 0)) for img in edv_imgs]
    esv_masks = [mask.numpy() for mask in esv_masks]
    edv_masks = [mask.numpy() for mask in edv_masks]
    esv_rgba_masks = [np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32) for mask in esv_masks]
    edv_rgba_masks = [np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32) for mask in edv_masks]
    for i, (edv_mask, esv_mask) in enumerate(zip(edv_masks, esv_masks)):
        esv_rgba_masks[i][esv_mask > 0] = [0, 0, 1, alpha]
        edv_rgba_masks[i][edv_mask > 0] = [0, 0, 1, alpha]

        esv_rgba_masks[i][esv_mask == 0] = [0, 0, 0, 0]
        edv_rgba_masks[i][edv_mask == 0] = [0, 0, 0, 0]
    fig, axes = plt.subplots(2, len(esv_imgs), figsize=(20, 20))
    plt.subplots_adjust(wspace=0, hspace=-0.5)
    for i, (esv_img, edv_img, esv_mask, edv_mask, filename) in enumerate(zip(esv_imgs, edv_imgs, esv_rgba_masks, edv_rgba_masks, batch['FileName'])):
        axes[0, i].imshow(esv_img, cmap='gray')
        axes[0, i].imshow(esv_mask, alpha=0.5)
        axes[0, i].axis('off')
        axes[0, i].set_title(filename)
        axes[1, i].imshow(edv_img, cmap='gray')
        axes[1, i].imshow(edv_mask, alpha=0.5)
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
        self.esv_idx = None
        self.edv_idx = None
        self.esv_poly = None
        self.edv_poly = None
        self.esv_frame = None
        self.edv_frame = None
        self.esv_mask = None
        self.edv_mask = None

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        csv_file = os.path.join(root_dir, 'Database', 'training_database.csv')
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
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
                if self.transform:
                    if patient.esv_idx == index:
                        patient.esv_frame = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    elif patient.edv_idx == index:
                        patient.edv_frame = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
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

        return {'FileName': patient.patient_id,
            'frames': {'esv': patient.esv_frame,
                   'edv': patient.edv_frame},
        'masks': {'esv': patient.esv_mask,
                  'edv': patient.edv_mask}}

# def process_polygon(polygon_str):
    
#     # Implement your function to convert polygon string to a mask
#     pass

