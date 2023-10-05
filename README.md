# Face Mask Detection using Convolutional Neural Networks

This project is a simple implementation of a Convolutional Neural Network (CNN) for detecting whether a person is wearing a face mask or not. The model is trained on a dataset of images containing people with and without masks.

## Dataset

The dataset used for this project is the [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) by Omkar Gurav, which contains over 7,000 images of people with and without masks.

## Requirements

To run this project, you will need the following:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV
- scikit-learn

I've installed these dependencies using pip:

```
pip install tensorflow numpy matplotlib opencv-python scikit-learn
```

## Usage

To use this project, follow these steps:

1. Clone the repository:

```
git clone https://github.com/your-username/face-mask-detection.git
```

2. Download the dataset from the link provided above and extract it to the `data` directory.

3. Train the model:

```
python train.py
```

4. Test the model:

```
python test.py
```

5. Predict on a new image:

```
python predict.py /path/to/image.jpg
```

## Results

The model achieves an accuracy  94% on the test set.
