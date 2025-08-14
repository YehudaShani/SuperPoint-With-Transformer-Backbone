# SuperPoint + Transformer Feature Backbone
Yoni Mandel and Yehuda Shani 
- Technion ECE - Deep Learning Project  
- Spring 2025
# Topics 
- Introduction
  - Project Objective
  - Motivation
  - Previous Work
- Design
  - Structure
  - Data
  - Metrics for Evaluation
- Results
  - Inference on a few Examples
  - Metrics Comparisson
- Conclusion
- Future work
- How to run
  - Training
  - Running Inference
  - Export for Evaluation
- Ethical statement
- References

# Introduction
## Project Objective 
This project adapts and extends the MagicPoint and SuperPoint frameworks for keypoint detection and description.
We explore improvements by integrating a transformer-based feature backbone to enhance feature representation and robustness under challenging imaging conditions.
The system is evaluated using standard metrics such as repeatability and homography estimation correctness on the HPatches dataset.
## Motivation 
Feature detection and matching are core components in tasks like SLAM, 3D reconstruction, and image registration.
While SuperPoint has proven highly effective, we hypothesize that a transformer-based backbone can improve its generalization and performance in scenarios with large appearance changes or geometric distortions.
## Previous Work 
MagicPoint: CNN-based interest point detector trained on synthetic data using homography adaptation.
SuperPoint: Extends MagicPoint with a descriptor head for joint detection and description, trained in a self-supervised manner.

We build on these by:
Replacing or augmenting the CNN backbone with a transformer-based feature extractor.
Retaining the dual-head architecture for detection and description.
Keeping the self-supervised training pipeline with homography adaptation.

# Design
## Structure
Our pipeline consists of:

- Detection Stage: MagicPoint or SuperPoint detector head produces interest point heatmaps.
- Description Stage: SuperPoint descriptor head extracts descriptors for detected keypoints.

Evaluation Stage:

- Repeatability: Measures consistency of keypoint detection across viewpoint changes.
- Homography Estimation Correctness: Measures how well descriptors can be matched to estimate geometric transformations.

## Data
We use:

- Synthetic Shapes for pretraining the detector.
- HPatches for evaluation of detection and matching.
- Optionally COCO or other datasets for fine-tuning.

Data preprocessing includes:

- Image resizing and normalization.
- Homographic warping for augmentation.

## Metrics for Evaulation
- Repeatability (compute_repeatability.py):

- Ratio of matching keypoints between two images under known homography.

- Homography Estimation Correctness (compute_desc_eval.py):

- Percentage of matches leading to a correct homography within a pixel threshold.
# Results
##  Inference on a few Examples
<video src="tracks_first200_once.mp4" controls width="600"></video>
TODO - Insert some examples of inference using both of the models, compare
## Metrics Comparison
TODO - Insert tables comparing results of both techniques
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









