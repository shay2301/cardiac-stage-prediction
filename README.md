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
- **Source**: EchoNet-Dynamic dataset
- **Size**: Approximately 10,000 echocardiogram videos
- **Type**: 2D echocardiogram videos with expert annotations

### Model Architecture
- Implemented both U-Net and TransUNet architectures for 2D segmentation
- Custom data loaders and transformations for efficient training
- Post-processing pipeline using optical flow for temporal consistency
- Signal processing techniques (FFT and inverse FFT) for noise reduction and BPM range filtering

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

## Results
- Successfully developed a proof-of-concept system for cardiac stage prediction
- Achieved 65% accuracy within 2 frames absolute error
- Demonstrated feasibility of automated cardiac stage detection

## Technical Skills Demonstrated
- Deep Learning (PyTorch)
- Computer Vision (OpenCV)
- Signal Processing
- Cloud Computing (AWS)
- Data Pipeline Development
- Model Training and Evaluation
- Docker Containerization

## Future Improvements
- Expand dataset size for better generalization
- Further optimize temporal consistency
- Enhance post-processing pipeline
- Explore real-time processing capabilities

## Getting Started
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the EchoNet-Dynamic dataset
4. Follow the training instructions in the documentation

## Note
This project was developed as a proof-of-concept over a 6-month period at VIZ.AI. While the results are promising, there are opportunities for further improvement with additional data and development time.

## License
[Add appropriate license information]

## Contact
[Add your contact information] 