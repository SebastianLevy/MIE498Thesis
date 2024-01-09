
# 3D Vision and Deep Learning Based Robotic Grasping for Human-Robot Interactions

## Overview
This project presents a novel approach to robotic grasping in cluttered environments, combining 3D vision with deep learning. The system integrates Mask R-CNN, 3D Convolutional Neural Networks (CNNs), and a UR5 robotic arm to identify, grasp, and manipulate objects accurately and adaptively.

## Dependencies
- Python 3.x
- NumPy
- OpenCV
- Matplotlib
- psutil
- Open3D (optional for point cloud visualization)

## Dataset
The dataset includes RGB-D images of ten objects: apples, bananas, Pringles cans, Cheez-It crackers, coffee cups, Coke cans, chocolate Jello boxes, mustard bottles, pears, and strawberries.

## Modules
- `detection.py`: Handles image resizing, object detection, and masking.
- `point_cloud.py`: Creates point clouds from masked images and depth maps (Requires Open3D).
- `voxel2.py`, `voxel_grid_augmentation2.py`: Manages voxel grid creation and augmentation.
- `dir_pred.py`, `mode_pred.py`: Predicts grasping directionality and modes using 3D CNNs.
- `imagecapture.py` (not included in the provided script): Captures RGB-D images (optional).
  
## Key Functions
- `runCode(image, depth)`: Main function to process images and depth maps for object detection, masking, and point cloud creation.
- `MatplotlibClearMemory()`: Clears memory used by Matplotlib figures.

## Usage
1. Import necessary modules.
2. Load RGB and depth images of the target object.
3. Call `runCode(image, depth)` to process the images and obtain detection and point cloud outputs.
4. (Optional) Visualize point clouds using Open3D.
5. Use the outputs for further analysis or robotic arm control.

## Additional Information
- The project includes exploration of different network structures and learning rates for the CNN models.
- Grasping modes include pinch grasp and hold actions for different object types.
- Future work includes expanding the dataset, refining models, and exploring neural network-driven haptic sensors.

## References
Refer to the thesis report for detailed methodology, experimental setup, and references.
