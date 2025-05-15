import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn, sigmoid, from_numpy, no_grad, load
from torchvision import transforms
from Scripts.data_transforms import input_transform, target_transform, video_transform
from torch.utils.data import Dataset, DataLoader
from Scripts import unet
from Scripts import dataloader
from Scripts.dataloader import imshow
import os
from concurrent.futures import ThreadPoolExecutor

def large_connected_component(prediction):
    prediction = prediction.detach().cpu().numpy().squeeze().astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(prediction)
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    return (labels == largest_label).astype(np.uint8)

def exponential_moving_average(data, alpha=0.9):
    ema_data = [data[0]]
    for value in data[1:]:
        ema_value = alpha * value + (1 - alpha) * ema_data[-1]
        ema_data.append(ema_value)
    return ema_data

def calculate_centroid_and_radius(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return (mask.shape[1] // 2, mask.shape[0] // 2), min(mask.shape[0], mask.shape[1]) // 4
    largest_contour = max(contours, key=cv2.contourArea)
    (cX, cY), radius = cv2.minEnclosingCircle(largest_contour)
    return (int(cX), int(cY)), int(radius)

def apply_geometric_constraints(mask, expected_center, expected_radius):
    h, w = mask.shape
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - expected_center[0]) ** 2 + (y - expected_center[1]) ** 2)
    circular_mask = dist_from_center <= expected_radius
    corrected_mask = np.logical_and(mask, circular_mask).astype(np.uint8)
    return corrected_mask

def median_filter(mask, kernel_size=5):
    return cv2.medianBlur(mask, kernel_size)

def process_patient(count, patient_id, frames, fps):
    print('{}) Processing patient {}'.format(count, patient_id))
    output_video_path = os.path.join(root_dir, 'EchoNet-Dynamic', 'Segmentation', 'output_{}.mp4'.format(patient_id))
    output_gif_path = os.path.join(root_dir, 'EchoNet-Dynamic', 'Segmentation', 'output_{}.gif'.format(patient_id))
    out = cv2.VideoWriter(output_video_path, fourcc, fps.item(), (112, 112))  # adjust frame size and fps as needed
    frames_indices_list = []
    mask_volumes = []
    gif_frames = []
    frames = frames[~(frames.reshape(frames.shape[0], -1).all(axis=1) == 0)]

    if not out.isOpened():
        print("Error: VideoWriter not opened. Check output path and fourcc codec.")
    else:
        for frame_idx in range(frames.shape[0]):
            frame = frames[frame_idx].to(device)
            frames_indices_list.append(frame_idx)
            tensor_frame = frame.unsqueeze(0)
            prediction = sigmoid(saved_model(tensor_frame)) > threshold
            prediction = large_connected_component(prediction)
            prediction = (prediction * 255).astype(np.uint8)
            frame = frame.detach().cpu().numpy()
            frame = (((frame - frame.min()) / (frame.max() - frame.min())) * 255).astype(np.uint8).squeeze()
            if frame.ndim == 2:  # if frame is grayscale, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if np.any(prediction):
                prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))
                mask_volume = np.count_nonzero(prediction)
                mask_volumes.append(mask_volume)
                green = np.zeros_like(frame)
                green[..., 1] = 255  # Set green channel to maximum
                green_prediction = cv2.bitwise_and(green, green, mask=prediction)
                overlay = cv2.addWeighted(frame, 1, green_prediction, 0.8, 0)
                out.write(overlay)
                gif_frames.append(overlay)
        out.release()
    smoothed_mask_volumes = exponential_moving_average(mask_volumes)
    time = np.arange(len(smoothed_mask_volumes)) / fps.item()
    plt.plot(time, smoothed_mask_volumes)
    plt.title('Mask Volume Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume')
    plt.savefig(os.path.join(root_dir, 'EchoNet-Dynamic', 'Segmentation', 'volume_plot_{}.png'.format(patient_id)))
    plt.clf()
    df_list.append(pd.DataFrame({'Patient': patient_id, 'Time (s)': time, 'Frame': frames_indices_list, 'Volume': smoothed_mask_volumes}))
    imageio.mimsave(output_gif_path, gif_frames, fps=fps.item())

if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    csv_full_path = os.path.join(root_dir, 'Database', 'training_database.csv')

    batch_size = 32
    threshold = 0.9

    full_video_dataset = dataloader.FullVideoDataset(root_dir=root_dir, input_transform=video_transform)
    full_video_dataloader = DataLoader(full_video_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load saved model
    saved_model = unet.UNet(n_channels=1, n_classes=1).to(device)
    saved_model.load_state_dict(load(os.path.join(root_dir, 'models', 'model_dice_6epochs.pth'), map_location=device))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'

    count = 1
    df_list = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, batch in enumerate(full_video_dataloader):
            for j, (patient_id, frames, fps) in enumerate(zip(batch['patient_id'], batch['frames'], batch['fps'])):
                futures.append(executor.submit(process_patient, count, patient_id, frames, fps))
                count += 1
        for future in futures:
            future.result()

    df = pd.concat(df_list)
    df.to_csv(os.path.join(root_dir, 'EchoNet-Dynamic', 'Segmentation', 'volume_data.csv'), index=False)
