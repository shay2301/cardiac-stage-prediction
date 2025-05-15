# Cardiac Stage Prediction from Echocardiograms

## Project Overview
This project, developed during my time at VIZ.AI, focuses on identifying systolic and diastolic end stages in echocardiograms using deep learning techniques. The goal was to accurately detect the end-diastolic volume (EDV) and end-systolic volume (ESV) from echocardiogram videos, which are crucial measurements in cardiac assessment.

### Key Challenges Addressed
- Development of an efficient 2D segmentation model for cardiac chamber detection
- Implementation of temporal data processing to ensure smooth and accurate volume measurements
- Handling of video data with varying quality and characteristics
- Post-processing of segmentation results to maintain temporal consistency

## Technical Implementation

### Tech Stack
- **Deep Learning Framework**: PyTorch
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Cloud Computing**: AWS EC2 (for training)

### Dataset
- **Source**: [EchoNet-Dynamic dataset](https://echonet.github.io/dynamic/) - A large-scale dataset of echocardiogram videos from Stanford University
- **Size**: Approximately 10,000 echocardiogram videos
- **Type**: 2D echocardiogram videos with expert annotations
- **Details**: Contains apical-4-chamber echocardiography videos with expert tracings of the left ventricle at end-systole and end-diastole

### Model Architecture
- Implemented both U-Net and TransUNet architectures for 2D segmentation
- Custom data loaders and transformations for efficient training
- Post-processing pipeline using optical flow for temporal consistency
- Signal processing techniques (FFT and inverse FFT) for noise reduction and BPM range filtering

### Technical Challenges and Solutions

#### 1. Temporal Consistency
- Implemented optical flow to ensure smooth transitions between frames
- Developed a post-processing pipeline to maintain temporal consistency in segmentation
- Created custom loss functions to penalize unrealistic temporal changes

#### 2. Signal Processing
- Applied FFT and inverse FFT to filter out noise and artifacts
- Implemented BPM range filtering to focus on relevant cardiac cycles
- Developed custom signal processing pipeline for volume calculation

#### 3. Data Processing
- Created efficient video data loaders for handling large echocardiogram datasets
- Implemented balanced dataset splitting to ensure representative training
- Developed robust data augmentation pipeline for improved model generalization

## Project Structure
```
├── models/           # Model architectures (U-Net, TransUNet)
├── Scripts/          # Data processing and utility scripts
│   ├── dataloader.py
│   ├── transformations.py
│   └── segmentation.py
├── Database/         # Patient data management
├── EDA/             # Exploratory data analysis notebooks
└── Challenges/      # Project challenges and solutions
```

## Key Technical Achievements

### 1. Data Processing Pipeline
- Developed custom data loaders for efficient video processing
- Implemented balanced train/test/validation splits following EchoNet research methodology
- Created robust data augmentation pipeline

### 2. Model Development
- Trained models from scratch using PyTorch
- Implemented both U-Net and TransUNet architectures
- Achieved 65% accuracy within 2 frames absolute error

### 3. Signal Processing
- Implemented FFT and inverse FFT for noise reduction
- Developed BPM range filtering for signal cleaning
- Used optical flow for temporal consistency in segmentation

### 4. Cloud Integration
- Set up training pipeline on AWS EC2
- Implemented efficient data transfer and processing
- Created Docker containerization for reproducible environments

## Results and Visualizations

### Model Performance
- Achieved 65% accuracy within 2 frames absolute error for stage detection
- Successfully segmented left ventricle with high precision
- Demonstrated robust performance across different video qualities

### Visual Results
![Cardiac Stage Prediction](animation.gif)
*Example of cardiac stage prediction showing segmentation and stage detection over time*

### Key Metrics
- Segmentation IoU (Intersection over Union): [Add specific metric if available]
- Stage Detection Accuracy: 65% within 2 frames
- Processing Speed: [Add if available]
- Validation Results: [Add if available]

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shay2301/cardiac-stage-prediction.git
   cd cardiac-stage-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the EchoNet-Dynamic dataset:
   - Visit [EchoNet-Dynamic](https://echonet.github.io/dynamic/)
   - Follow their data access instructions
   - Place the dataset in the appropriate directory

### Training
1. Prepare your data:
   ```bash
   python Scripts/prepare_data.py
   ```

2. Train the model:
   ```bash
   python train.py --config configs/train_config.yaml
   ```

### Evaluation
1. Run evaluation on test set:
   ```bash
   python eval.py --model_path path/to/model --data_path path/to/test/data
   ```

2. Generate visualizations:
   ```bash
   python full_video_eval.py --video_path path/to/video --model_path path/to/model
   ```

## Project Timeline
- **Month 1**: Data exploration and preprocessing pipeline development
- **Month 2**: Initial model architecture implementation and training
- **Month 3**: Signal processing and temporal consistency improvements
- **Month 4**: Model optimization and performance tuning
- **Month 5**: Post-processing pipeline development
- **Month 6**: Final evaluation and documentation

## Technical Skills Demonstrated
- Deep Learning (PyTorch)
- Computer Vision (OpenCV)
- Signal Processing
- Cloud Computing (AWS)
- Data Pipeline Development
- Model Training and Evaluation
- Docker Containerization

## Future Improvements
- Test the model on real-world clinical data to validate performance in practical settings
- Further optimize temporal consistency
- Enhance post-processing pipeline
- Explore real-time processing capabilities

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The EchoNet-Dynamic dataset used in this project is subject to its own [Research Use Agreement](https://echonet.github.io/dynamic/).

## Contact
Shay Levi  
Email: shay230@gmail.com  
LinkedIn: [Shay Levi](https://www.linkedin.com/in/shaylevi/)  
GitHub: [shay2301](https://github.com/shay2301/shay2301) 