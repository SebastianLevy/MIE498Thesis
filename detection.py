import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import cv2
import open3d as o3d

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg


np.set_printoptions(threshold=sys.maxsize)

# Root directory of the project
ROOT_DIR = os.path.abspath("D:/School/UofT/Research/MaskRCNN")
# 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import final

#matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights
WEIGHTS_PATH = "D:/School/UofT/Research/MaskRCNN/logs/object20230225T1002/mask_rcnn_thesis_0100.h5"  # TODO: update this path


config = final.CustomConfig()
CUSTOM_DIR = os.path.join(ROOT_DIR, "main/dataset")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Load validation dataset
dataset = final.CustomDataset()
dataset.load_custom(CUSTOM_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# load the last model you trained
weights_path = "D:/School/UofT/Research/MaskRCNN/logs/object20230225T1002/mask_rcnn_thesis_0100.h5"
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

def image_resize2(img):
    img = cv2.resize(img, (1024,1024))
    return img


def image_resize(img):
    # Load the original image
    #img = cv2.imread('D:/School/UofT/Research/images/large_dataset/rgb/img{}.png'.format(id))
    # Define the new size
    new_size = (1024, 574)
    # Resize the image while preserving the aspect ratio
    resized = img #cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    # Create a new black image of the target size
    black_image = np.zeros((new_size[0], new_size[0], 3), np.uint8)

    # Determine the size of each image
    height1, width1, _ = black_image.shape
    height2, width2, _ = resized.shape
    # Calculate the position to paste the second image
    x_offset = int((width1 - width2) / 2)
    y_offset = int((height1 - height2) / 2)
    # Resize the second image to fit into the first image
    resized_image2 = cv2.resize(resized, (width2, height2))
    # Paste the resized second image onto the first image
    black_image[y_offset:y_offset+height2, x_offset:x_offset+width2] = resized_image2
    # cv2.imshow("New Image", black_image)
    # cv2.waitKey(7000)
    # cv2.destroyAllWindows()
    return(black_image)

def depth_resize(foreground):
    #foreground = np.load('D:/School/UofT/Research/MaskRCNN/main/Mask_RCNN/images/large_dataset/depth/img{}.npy'.format(id))
    # Create a black 1024x1024 numpy array
    background = np.zeros((1024, 1024), dtype=np.uint16)

    # Calculate the offset to paste the foreground in the center of the background
    offset_x = int((background.shape[1] - foreground.shape[1]) / 2)
    offset_y = int((background.shape[0] - foreground.shape[0]) / 2)

    # Define the slice to paste the foreground in the center of the background
    slice_x = slice(offset_x, offset_x + foreground.shape[1])
    slice_y = slice(offset_y, offset_y + foreground.shape[0])

    # Paste the foreground into the background
    background[slice_y, slice_x] = foreground
    # cv2.imshow('Result', background)
    # cv2.waitKey(15000)
    # cv2.destroyAllWindows()
    log('depth array',background)
    return(background)



def detectMask(image):
    #image = mpimg.imread('D:/School/UofT/Research/MaskRCNN/main/Mask_RCNN/images/images/img{}.png'.format(image_id))
    #image = cv2.resize(image,(1024,1024), interpolation=cv2.INTER_AREA)
    results = model.detect([image], verbose=1)
    r = results[0]
    ax = get_ax(1)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax,
                                title="Predictions")
    mrcnn = model.run_graph([image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ])
    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]
    print("{} detection: {}".format(
        det_count, np.array(dataset.class_names)[det_class_ids]))
    # Masks

    det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c] 
                                for i, c in enumerate(det_class_ids)])
    det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                        for i, m in enumerate(det_mask_specific)])
    #log("det_mask_specific", det_mask_specific)
    #log("det_masks", det_masks)
    if det_count != 0:
        det_masks = det_masks.astype(np.uint8)[0]
    return(det_masks, det_count, np.array(dataset.class_names)[det_class_ids])
    

        

def depth_mask(mask, image, depth):

    # Convert the depth image to float32
    # log('rbg_image',image)
    # log('mask_array',mask)
    # Apply the mask to the RGB and depth images
    image = cv2.bitwise_and(image, image, mask=mask)
    depth = cv2.bitwise_and(depth, depth, mask=mask)
    # Convert the depth image back to uint8
    # depth = depth.astype(np.uint8)
    return(image, depth)

def depth_to_pointcloud(depth_map, fx, fy, cx, cy):
    """
    Convert a depth map to a 3D point cloud.
    
    Args:
        depth_map: numpy array representing the depth map
        fx: focal length in the x direction
        fy: focal length in the y direction
        cx: x coordinate of the principal point
        cy: y coordinate of the principal point
        
    Returns:
        numpy array representing the 3D point cloud
    """
    # Compute the x and y coordinates of the pixels
    x, y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
    
    # Compute the corresponding x, y, and z coordinates of the points
    z = depth_map.flatten()
    x = (x.flatten() - cx) * z / fx
    y = (y.flatten() - cy) * z / fy
    
    # Stack the x, y, and z coordinates into a 3D array
    point_cloud = np.vstack((x, y, z)).T
    
    return point_cloud

def i_mask(mask):
    contours,_ = cv2.findContours(mask.copy(), 1, 1) # not copying here will throw an error
    rect = cv2.minAreaRect(contours[0]) # basically you can feed this rect into your classifier
    new_mask = np.zeros((1024, 1024), np.uint8)
    (x,y),(w,h), a = rect # a - angle
    cc = (int(x),int(y))
    # print(cc)
    hypo = np.sqrt((w/2)**2+(h/2)**2)
    radius = int(hypo*1.5)
    if radius < 150:
        radius = 150
    colour = (255,255,255)
    new_mask = cv2.circle(new_mask,cc,radius,colour,-1)
    inv_mask = cv2.bitwise_not(mask*255)
    circle_mask = cv2.bitwise_and(inv_mask, inv_mask, mask=new_mask)    
    # print(np.info(mask))
    #cv2.imshow('mask', mask.copy())
    # print(np.info(new_mask))
    # cv2.imshow('new_mask', new_mask)
    # print(np.info(inv_mask))
    #cv2.imshow('inv_mask', inv_mask)
    #cv2.imshow('circle_mask', circle_mask)
    #cv2.waitKey(0)
    return(circle_mask, int(x), int(y)) #replace with circle mask


def circle_mask(mask):
    contours,_ = cv2.findContours(mask.copy(), 1, 1) # not copying here will throw an error
    rect = cv2.minAreaRect(contours[0]) # basically you can feed this rect into your classifier
    new_mask = np.zeros((1024, 1024), np.uint8)
    (x,y),(w,h), a = rect # a - angle
    cc = (int(x),int(y))
    hypo = np.sqrt((w/2)**2+(h/2)**2)
    radius = int(hypo*1.5)
    if radius < 150:
        radius = 150
    colour = (255,255,255)
    new_mask = cv2.circle(new_mask,cc,radius,colour,-1)
    inv_mask = cv2.bitwise_not(mask*255)
    circle_mask = cv2.bitwise_and(inv_mask, inv_mask, mask=new_mask)    
    # print(np.info(mask))
    # cv2.imshow('mask', mask)
    # print(np.info(new_mask))
    # cv2.imshow('new_mask', new_mask)
    # print(np.info(inv_mask))
    # cv2.imshow('inv_mask', inv_mask)
    # cv2.imshow('circle_mask', circle_mask)
    # cv2.waitKey(0)
    return(new_mask) #replace with circle mask


temp = []
files = []

f = open("D:/School/UofT/Research/MaskRCNN/final_list.txt", "r")
for x in f:
    temp.append(int(x[:-1]))
for i in range(559):
     if temp.count(i) == 0:
          files.append(i)


'''
for i in range(1):
    print(f'Image {i}')
    image_id = 0
    image = image_resize(image_id)
    depth = depth_resize(image_id)
    mask, det_count = detectMask(image)
    if det_count == 0:
        with open("final_list2.txt", "a") as file:
            file.write(str(image_id) + "\n")
        continue
    mask_image, mask_depth  = depth_mask(mask, image, depth)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    f1 = plt.figure()
    plt.imshow(depth, cmap='gray')
    plt.show()
    inv_mask, x, y = i_mask(mask)
    inv_mask_image, inv_mask_depth  = depth_mask(inv_mask, image, depth)
    c_mask = circle_mask(mask)
    xyz = [x,y,depth[x,y]]
    print(xyz)
    c_mask_image, c_mask_depth  = depth_mask(c_mask, image, depth)


    cv2.imwrite('/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/MaskRCNN/main/Mask_RCNN/images/grasping_cnn/dataset/m_rgb/img{}.png'.format(image_id), mask_image)
    np.save('/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/MaskRCNN/main/Mask_RCNN/images/grasping_cnn/dataset/m_depth/img{}.npy'.format(image_id), mask_depth)
    cv2.imwrite('/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/MaskRCNN/main/Mask_RCNN/images/grasping_cnn/dataset/inv_rgb/img{}.png'.format(image_id), inv_mask_image)
    np.save('/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/MaskRCNN/main/Mask_RCNN/images/grasping_cnn/dataset/inv_depth/img{}.npy'.format(image_id), inv_mask_depth)
    cv2.imwrite('/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/MaskRCNN/main/Mask_RCNN/images/grasping_cnn/dataset/c_rgb/img{}.png'.format(image_id), c_mask_image)
    np.save('/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/MaskRCNN/main/Mask_RCNN/images/grasping_cnn/dataset/c_depth/img{}.npy'.format(image_id), c_mask_depth)




'''
