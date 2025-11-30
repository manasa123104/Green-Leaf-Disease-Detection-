# 🌿 Green Leaf Disease Detection using CNN Algorithm

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-green)](https://www.tensorflow.org/)
[![Raspberry Pi](https://img.shields.io/badge/Platform-Raspberry%20Pi-red)](https://www.raspberrypi.org/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Image%20Classification-orange)](https://opencv.org/)

An intelligent **plant disease detection system** using Convolutional Neural Networks (CNN) to identify and classify diseases in green leaves. This project implements deep learning algorithms for automated agricultural disease diagnosis.

## 🎯 Project Overview

This system uses **CNN (Convolutional Neural Network)** algorithms to automatically detect and classify diseases in plant leaves. The project was developed as a major project at **CVR College of Engineering** and can be deployed on **Raspberry Pi** for real-time disease detection in agricultural settings.

## ✨ Features

### 🔬 Disease Detection
- **Automated Classification**: CNN-based image classification for plant diseases
- **Real-time Detection**: Fast inference for immediate disease identification
- **Multiple Disease Types**: Classification of various leaf diseases
- **High Accuracy**: Deep learning model trained on plant disease datasets

### 🖼️ Image Processing
- **Image Preprocessing**: Automated image enhancement and normalization
- **Feature Extraction**: CNN-based feature learning from leaf images
- **Classification**: Multi-class disease classification

### 🍓 Raspberry Pi Deployment
- **Edge Computing**: Deployable on Raspberry Pi for field use
- **Portable Solution**: Lightweight system for agricultural applications
- **Real-time Processing**: On-device inference without cloud dependency

## 🛠️ Technology Stack

### Core Technologies
- **Python** - Primary programming language
- **Deep Learning Framework** - TensorFlow/Keras or PyTorch
- **Computer Vision** - OpenCV for image processing
- **CNN Architecture** - Convolutional Neural Networks

### Hardware
- **Raspberry Pi** - Edge computing device
- **Camera Module** - For image capture
- **Display** - For results visualization

### Libraries & Tools
- **NumPy** - Numerical computations
- **Pillow/PIL** - Image processing
- **Matplotlib** - Visualization
- **Scikit-learn** - Model evaluation

## 📁 Project Structure

```
Green-Leaf-Disease-Detection-/
├── Major Project Final.pdf              # Project documentation and report
├── remotesensing-13-04218-v2.pdf        # Research paper reference
├── models/                              # Trained CNN models
│   ├── disease_detection_model.h5
│   └── model_weights.h5
├── dataset/                             # Training dataset
│   ├── healthy_leaves/
│   ├── diseased_leaves/
│   └── test_images/
├── src/                                 # Source code
│   ├── train_model.py                  # Model training script
│   ├── predict.py                      # Prediction script
│   ├── preprocess.py                   # Image preprocessing
│   └── cnn_model.py                    # CNN architecture
├── raspberry_pi/                        # Raspberry Pi deployment
│   ├── capture_image.py
│   ├── run_detection.py
│   └── requirements.txt
├── notebooks/                          # Jupyter notebooks
│   └── model_development.ipynb
└── README.md                           # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.7 or higher
python --version

# Required packages
pip install tensorflow keras opencv-python numpy pillow matplotlib scikit-learn
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/manasa123104/Green-Leaf-Disease-Detection-.git
   cd Green-Leaf-Disease-Detection-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare dataset**
   - Organize images into folders by disease type
   - Place training images in `dataset/train/`
   - Place test images in `dataset/test/`

### Usage

#### Training the Model

```bash
python src/train_model.py --dataset dataset/train --epochs 50 --batch_size 32
```

#### Making Predictions

```bash
python src/predict.py --image path/to/leaf_image.jpg --model models/disease_detection_model.h5
```

#### Raspberry Pi Deployment

```bash
cd raspberry_pi
python run_detection.py
```

## 🧠 CNN Architecture

The system uses a Convolutional Neural Network with the following architecture:

```
Input Layer (Image)
    ↓
Convolutional Layers (Feature Extraction)
    ↓
Pooling Layers (Dimensionality Reduction)
    ↓
Fully Connected Layers (Classification)
    ↓
Output Layer (Disease Classes)
```

### Model Features
- **Transfer Learning**: Pre-trained models for better accuracy
- **Data Augmentation**: Enhanced training with image transformations
- **Dropout Regularization**: Prevents overfitting
- **Batch Normalization**: Stabilizes training

## 📊 Dataset

The model is trained on plant disease datasets containing:
- **Healthy Leaves**: Images of healthy plant leaves
- **Diseased Leaves**: Images of leaves with various diseases
- **Multiple Plant Types**: Different plant species
- **Various Disease Types**: Different disease categories

### Data Preprocessing
- Image resizing and normalization
- Data augmentation (rotation, flipping, scaling)
- Train/validation/test split
- Label encoding

## 🔬 Model Performance

- **Accuracy**: High classification accuracy on test dataset
- **Precision**: Accurate disease identification
- **Recall**: Comprehensive disease detection
- **F1-Score**: Balanced performance metrics

## 📱 Raspberry Pi Deployment

### Setup on Raspberry Pi

1. **Install dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install tensorflow-lite opencv-python
   ```

2. **Connect camera module**
   - Attach Raspberry Pi Camera Module
   - Enable camera interface in settings

3. **Run detection**
   ```bash
   python3 raspberry_pi/run_detection.py
   ```

### Features on Raspberry Pi
- Real-time image capture
- On-device inference
- Display results on screen
- Save detection results

## 📈 Results & Evaluation

The CNN model demonstrates:
- ✅ High accuracy in disease classification
- ✅ Fast inference time suitable for real-time use
- ✅ Robust performance on various leaf conditions
- ✅ Effective feature extraction from leaf images

## 📚 Research References

- **Major Project Final.pdf**: Complete project documentation
- **remotesensing-13-04218-v2.pdf**: Remote sensing research paper reference

## 🎓 Project Details

- **Institution**: CVR College of Engineering
- **Project Type**: Major Project
- **Domain**: Computer Vision, Deep Learning, Agriculture
- **Application**: Agricultural Disease Detection

## 🔄 Workflow

```
Image Capture → Preprocessing → CNN Feature Extraction → Classification → Disease Identification
```

## 🛡️ Use Cases

1. **Agricultural Monitoring**: Automated disease detection in crops
2. **Early Detection**: Identify diseases before visible symptoms worsen
3. **Precision Agriculture**: Targeted treatment based on disease type
4. **Research**: Plant pathology research and analysis
5. **Field Deployment**: Portable disease detection in agricultural fields

## 🚧 Future Enhancements

- [ ] Support for more plant species
- [ ] Mobile app integration
- [ ] Cloud-based model updates
- [ ] Multi-disease detection
- [ ] Severity assessment
- [ ] Treatment recommendations
- [ ] Integration with IoT sensors
- [ ] Real-time monitoring dashboard

## 📝 Model Training Tips

1. **Data Quality**: Ensure high-quality, labeled images
2. **Augmentation**: Use data augmentation to increase dataset size
3. **Hyperparameter Tuning**: Optimize learning rate, batch size, epochs
4. **Transfer Learning**: Use pre-trained models for better results
5. **Regularization**: Apply dropout and batch normalization

## 🔍 Testing

```bash
# Test on single image
python src/predict.py --image test_image.jpg

# Evaluate on test dataset
python src/evaluate.py --test_dir dataset/test
```

## 📊 Performance Metrics

- Model accuracy on test dataset
- Inference time per image
- Memory usage on Raspberry Pi
- Model size optimization

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Model architecture optimization
- Dataset expansion
- Mobile app development
- Documentation improvements
- Performance optimization

## 📄 License

This project is for educational and research purposes.

## 👤 Author

**Manasa Rajagopal Madabushi**
- GitHub: [@manasa123104](https://github.com/manasa123104)
- Portfolio: [https://manasa123104.github.io/Manasa-portfolio/](https://manasa123104.github.io/Manasa-portfolio/)

## 🔗 Related Projects

- Plant Disease Classification
- Agricultural AI Systems
- Computer Vision in Agriculture
- Edge AI Applications

## 📚 Learning Outcomes

This project demonstrates:
- ✅ Deep learning model development
- ✅ CNN architecture design and implementation
- ✅ Image classification techniques
- ✅ Edge computing deployment
- ✅ Agricultural AI applications
- ✅ Real-time inference optimization

## 🌱 Impact

This system can help:
- **Farmers**: Early disease detection for better crop management
- **Researchers**: Automated plant disease analysis
- **Agriculture Industry**: Precision agriculture solutions
- **Food Security**: Improved crop yield through early intervention

---

**Built with ❤️ for Agricultural Innovation**

**Note**: This project was developed as part of academic coursework at CVR College of Engineering, demonstrating practical application of deep learning in agriculture.

