import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from torch import sigmoid, from_numpy, no_grad, load
from torchvision import transforms
from Scripts.data_transforms import video_transform
from torch.utils.data import Dataset, DataLoader
from Scripts.models import transunet
from Scripts import dataloader
from Scripts.dataloader import imshow
import os
import psutil
import time
import gc
import pynvml

def log_cpu_memory_usage():
    memory_info = psutil.virtual_memory()
    used_memory = memory_info.used / (1024 * 1024)
    total_memory = memory_info.total / (1024 * 1024)
    available_memory = memory_info.available / (1024 * 1024)
    print(f"CPU Memory Used: {used_memory:.2f} MB")
    print(f"CPU Memory Available: {available_memory:.2f} MB")
    print(f"CPU Memory Total: {total_memory:.2f} MB")

def log_gpu_memory_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming a single GPU, index 0
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_memory = info.used / (1024 * 1024)
    total_memory = info.total / (1024 * 1024)
    free_memory = info.free / (1024 * 1024)
    print(f"GPU Memory Used: {used_memory:.2f} MB")
    print(f"GPU Memory Free: {free_memory:.2f} MB")
    print(f"GPU Memory Total: {total_memory:.2f} MB")

def apply_temporal_consistency(current_pred, prev_predictions, num_prev_frames=5, alpha=0.7):
    prev_predictions = prev_predictions[-num_prev_frames:]
    
    if len(prev_predictions) == 0:
        return current_pred
    
    weights = np.exp(-np.arange(len(prev_predictions)) / 2)  # Exponential decay weights
    weights = weights / np.sum(weights)
    avg_prev_pred = np.average(prev_predictions, axis=0, weights=weights)
    
    consistent_pred = (alpha * current_pred + (1 - alpha) * avg_prev_pred) > 0.5
    
    return consistent_pred.astype(np.uint8)

def large_connected_component(prediction):
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    prediction = prediction.squeeze().astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(prediction)
    if num_labels > 1:
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        return (labels == largest_label).astype(np.uint8)
    else:
        return prediction

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

def compute_optical_flow(prev_frame, curr_frame):
    # Convert PyTorch tensors to NumPy arrays
    prev_frame = prev_frame.squeeze().cpu().numpy().astype(np.uint8)
    curr_frame = curr_frame.squeeze().cpu().numpy().astype(np.uint8)
    
    # Ensure the frames are 2D (grayscale)
    if prev_frame.ndim == 3:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    if curr_frame.ndim == 3:
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def warp_frame(frame, flow):
    # Ensure frame is a NumPy array
    if isinstance(frame, torch.Tensor):
        frame = frame.squeeze().cpu().numpy()
    
    h, w = frame.shape[:2]
    flow_map = -flow
    flow_map[:,:,0] += np.arange(w)
    flow_map[:,:,1] += np.arange(h)[:,np.newaxis]
    warped = cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)
    return warped

def process_patient(saved_model, count, patient_id, frames, fps, root_dir):
    print(f'{count}) Processing patient {patient_id}')
    output_video_path = os.path.join(root_dir, 'EchoNet-Dynamic', 'Segmentation', f'output_{patient_id}.mp4')
    output_gif_path = os.path.join(root_dir, 'EchoNet-Dynamic', 'Segmentation', f'output_{patient_id}.gif')
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps.item(), (224, 224))
    
    frames_indices_list = []
    mask_volumes = []
    gif_frames = []
    frames = frames[~(frames.reshape(frames.shape[0], -1).all(axis=1) == 0)]

    if not out.isOpened():
        print(f"Error: VideoWriter not opened for patient {patient_id}. Check output path and fourcc codec.")
        return

    prev_frame = None
    prev_prediction = None
    
    try:
        for frame_idx in range(frames.shape[0]):
            frame = frames[frame_idx].to(device)
            tensor_frame = frame.unsqueeze(0)
                        
            # Generate new segmentation for each frame
            with torch.no_grad():
                raw_prediction = saved_model(tensor_frame)
            prediction = sigmoid(raw_prediction)
            prediction = F.interpolate(prediction, size=(224, 224), mode='bilinear', align_corners=False)
            prediction = (prediction > threshold).float().cpu().numpy().squeeze()
            
            # Apply large connected component
            prediction = large_connected_component(prediction)
            
            # Apply temporal consistency if not the first frame
            if prev_frame is not None and prev_prediction is not None:
                flow = compute_optical_flow(prev_frame, frame)
                warped_prev_prediction = warp_frame(prev_prediction, flow)
                
                # Blend current prediction with warped previous prediction
                alpha = 0.7  # Adjust this value to control temporal consistency
                prediction = alpha * prediction + (1 - alpha) * warped_prev_prediction
                prediction = (prediction > 0.5).astype(np.uint8)
            
            # Convert prediction to 8-bit for visualization
            prediction_vis = (prediction * 255).astype(np.uint8)
            
            # Prepare frame for visualization
            frame_np = frame.detach().cpu().numpy()
            frame_np = (((frame_np - frame_np.min()) / (frame_np.max() - frame_np.min())) * 255).astype(np.uint8).squeeze()
            if frame_np.ndim == 2:
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
            
            # Create overlay
            if np.any(prediction):
                mask_volume = np.count_nonzero(prediction)
                mask_volumes.append(mask_volume)
                green = np.zeros_like(frame_np)
                green[..., 1] = 255
                green_prediction = cv2.bitwise_and(green, green, mask=prediction_vis)
                overlay = cv2.addWeighted(frame_np, 1, green_prediction, 0.5, 0)
                out.write(overlay)
                gif_frames.append(overlay)
            else:
                out.write(frame_np)
                gif_frames.append(frame_np)

            frames_indices_list.append(frame_idx)
            
            prev_frame = frame.cpu()
            prev_prediction = prediction

    except Exception as e:
        print(f"Exception occurred while processing patient {patient_id}: {str(e)}")
        print(f"Error occurred at line: {e.__traceback__.tb_lineno}")
        import traceback
        traceback.print_exc()
    finally:
        out.release()

        if mask_volumes:
            smoothed_mask_volumes = exponential_moving_average(mask_volumes)
            time = np.arange(len(smoothed_mask_volumes)) / fps.item()
            
            if len(time) != len(frames_indices_list) or len(time) != len(smoothed_mask_volumes):
                print("Mismatch in lengths detected! Truncating lists to the shortest length.")
                min_length = min(len(time), len(frames_indices_list), len(smoothed_mask_volumes))
                time = time[:min_length]
                frames_indices_list = frames_indices_list[:min_length]
                smoothed_mask_volumes = smoothed_mask_volumes[:min_length]
            
            plt.plot(time, smoothed_mask_volumes)
            plt.title('Mask Volume Over Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Volume')
            plt.savefig(os.path.join(root_dir, 'EchoNet-Dynamic', 'Segmentation', f'volume_plot_{patient_id}.png'))
            plt.clf()
            
            df = pd.DataFrame({'Patient': patient_id, 'Time (s)': time, 'Frame': frames_indices_list, 'Volume': smoothed_mask_volumes})
            volume_data_csv_path = os.path.join(root_dir, 'EchoNet-Dynamic', 'Segmentation', 'volume_data.csv')
            if os.path.exists(volume_data_csv_path):
                df.to_csv(volume_data_csv_path, mode='a', index=False, header=False)
            else:
                df.to_csv(volume_data_csv_path, mode='w', index=False, header=True)
        else:
            print(f"No mask detected for patient {patient_id}")

        if gif_frames:
            if fps.item() > 50:
                imageio.mimsave(output_gif_path, gif_frames, fps=50)
            else:
                imageio.mimsave(output_gif_path, gif_frames, fps=fps.item())
        else:
            print(f"No frames processed for patient {patient_id}")

        # Explicitly delete variables to free memory
        del frames, mask_volumes, gif_frames, smoothed_mask_volumes, df, tensor_frame, prediction, frame, overlay
        gc.collect()

if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    csv_full_path = os.path.join(root_dir, 'Database', 'training_database.csv')

    batch_size = 2  # Reduced batch size to lower memory usage further
    threshold = 0.9

    full_video_dataset = dataloader.FullVideoDataset(root_dir=root_dir, transform=video_transform, resize_flag=True)
    full_video_dataloader = DataLoader(full_video_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, persistent_workers=False)  # Reduced num_workers further

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load saved model
    model_name = 'transunet_6epochs.pth'
    saved_model = transunet.TransUNet112(img_size=224, patch_size=8, num_classes=1, in_channels=1)
    saved_model.load_state_dict(load(os.path.join(root_dir, 'models', model_name), map_location=device))
    saved_model.to(device)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'

    # Initialize NVML for GPU memory logging
    pynvml.nvmlInit()

    count = 1

    # Load existing volume_data.csv to get the last processed patient
    volume_data_csv_path = os.path.join(root_dir, 'EchoNet-Dynamic', 'Segmentation', 'volume_data.csv')
    if os.path.exists(volume_data_csv_path):
        processed_patients_df = pd.read_csv(volume_data_csv_path)
        processed_patients = processed_patients_df['Patient'].unique()
        print(f"Resuming from the last processed patient. Total processed: {len(processed_patients)}")
    else:
        processed_patients = []
        print("No previous data found. Starting fresh.")

    try:
        for i, batch in enumerate(full_video_dataloader):
            for j, (patient_id, frames, fps) in enumerate(zip(batch['patient_id'], batch['frames'], batch['fps'])):
                count += 1
                if patient_id not in processed_patients:
                    process_patient(saved_model, count, patient_id, frames, fps, root_dir)
                    # log_cpu_memory_usage()
                    # log_gpu_memory_usage()
                else:
                    print(f"{count}) Skipping patient {patient_id} as it has already been processed.")
    finally:
        pynvml.nvmlShutdown()  # Shutdown NVML once after the loop
