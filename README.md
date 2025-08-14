# SuperPoint + Transformer Backbone
Yoni Mandel and Yehuda Shani 
- Technion ECE - Deep Learning Project 
- Spring 2025

# Introduction 
## Project Objective 
This project adapts and extends the MagicPoint and SuperPoint frameworks for keypoint detection. To enhance feature representation and improve robustness under challenging imaging conditions, we integrate a Transformer-based feature extraction backbone into the detection pipeline. As inter-frame point correspondence will be established through optical flow tracking, and due to limited GPU compute resources, the descriptor computation is omitted. The proposed system is evaluated on the EuRoC dataset, a widely used benchmark for Visual Inertial Odometry, using established performance metrics, including repeatability and homography estimation accuracy.
## Motivation 
Feature detection and matching are fundamental components of numerous computer vision tasks, including simultaneous localization and mapping (SLAM), 3D reconstruction, and image registration. While SuperPoint has demonstrated strong performance across a variety of conditions, we hypothesize that replacing its convolutional backbone with a transformer-based architecture can further improve generalization and robustness, particularly in scenarios involving substantial appearance variations or geometric distortions.
## Previous Work 
MagicPoint is a convolutional neural network (CNN)-based interest point detector trained entirely on synthetically generated data.
SuperPoint builds upon MagicPoint through a self-supervised learning strategy, employing homography adaptation to improve detection performance on real-world imagery without requiring manual annotations. 

# Design
## Structure
Encoder – The original SuperPoint encoder is a convolutional network that turns the input grayscale image into a smaller but more detailed feature map. It uses several convolution layers with ReLU activation and pooling, reducing the size by 8× while keeping important structures like corners and edges. This feature map is then used by the detector.

In our new design, the CNN is replaced with a Transformer-based encoder. The image is split into small patches, each turned into an embedding vector with position information. These go through transformer layers that use self-attention to connect features from all over the image, not just nearby pixels. This allows the network to handle big viewpoint changes, distortions, and difficult lighting better than the CNN. The final output is reshaped into a feature map for the detector, just like before.

The detector head takes the feature map from the encoder and predicts where keypoints are likely to be in the image. It uses a few small convolution layers to produce a heatmap. The original design is kept, where each cell of the heatmap has 65 values — one for each of the 8×8 positions inside the cell, plus one “no keypoint” option. A softmax is applied so the values become probabilities, then the map is reshaped and upsampled back to the original image size. The points with the highest probabilities are selected as the final keypoints.

## Data
The experimental pipeline utilizes the following datasets:

- Synthetic Shapes — employed for pretraining the keypoint detector, providing a controlled environment for learning fundamental geometric structures.

- EuRoC MAV Dataset — used for evaluating detection performance under realistic conditions relevant to visual-inertial odometry.

- COCO or other large-scale image datasets — optionally used for fine-tuning to improve generalization to diverse scenes.

Data Preprocessing - all images undergo the following preprocessing steps prior to model training or evaluation:

- Resizing and normalization — to standardize image dimensions and pixel intensity ranges across datasets.

- Homographic warping — applied as a data augmentation technique to simulate viewpoint changes and geometric transformations, enhancing robustness to real-world variations.

## Metrics for Evaulation
1. Repeatability -  Measures how consistently the same keypoints are detected in different views of the same scene.
2. Localization Error -  Average pixel distance between the projected ground-truth keypoint position and the detected keypoint position
3. Homography Estimation Accuracy / Correctness - Measures how accurately the detected keypoints can be used to estimate the geometric transformation between image pairs.
   
# Results
##  Inference on a few Examples
TODO - Insert some examples of inference using both of the models, compare

## Metrics Comparison
<video controls autoplay muted loop playsinline width="640">
  <source src="https://mandelyoni.github.io/SuperPoint-With-Transformer-Backbone/tracks_first200_once.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

# Conclusion

TODO - Insert conclusion of the project
# Future work
TODO - Insert future work
# How to run
## Clone the repository:
TODO
```
git clone https://our-git-repo
``` 
## Training
```
python train.py config/magic_point_syn_train.yaml  # For MagicPoint
python train.py config/superpoint_train.yaml       # For SuperPoint
```
## Running Inference:
```
python train.py inference_check.py # Change file path to correct image!
```
## Export for Evaluation:
```
python export_detections_repeatability.py 
python export_descriptors.py
```

# Ethical statement 
## Stakeholders: 
- Computer vision researchers and developers working on SLAM, AR/VR, and robotics.

- Companies deploying vision-based systems in navigation, mapping, or object tracking.

- The general public whose images or environments may be processed by such systems.
## Implications: 
Teams can optimize performance and race strategies, though over-reliance may limit human adaptability. Broadcasters and commentators can enhance analysis, enriching the fan experience with deeper insights.
## Ethical Considerations: 
There has to be clear boundaries on acceptable use, in general, since the amount of surveillance is growing expnoentially, models that can be implemented to retrieve data from them might result in privacy issues.
# References
TODO - Insert References









