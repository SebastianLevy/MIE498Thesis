#import imagecapture as ic
import detection as dt
import point_cloud as pc
import voxel2 as vx
import dir_pred as dp
import mode_pred as mp
import numpy as np
#import open3d as o3d
import cv2
import os
import psutil 
import numpy
import matplotlib
import matplotlib.pyplot as plt
import voxel_grid_augmentation2 as aug
import copy
import csv



# --IMAGE CAPTURE--
#intr, depth_intrin, image, depth, images_depth_colormap, depth_image_3d, aligned_depth_frame = ic.capture_rgbd_images()


def runCode(image,depth):
    # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # depth = np.rot90(depth, k=1)
    # --DETECTION--
    image = dt.image_resize(image)
    depth = dt.depth_resize(depth)
    mask, det_count, object= dt.detectMask(image)
    if det_count == 0:
        print('No Object Found')
        test = 0
        return None, None, None, None, test
    mask_image, mask_depth  = dt.depth_mask(mask, image, depth)
    cv2.imshow('image', mask_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    inv_mask, x, y = dt.i_mask(mask)
    inv_mask_image, inv_mask_depth  = dt.depth_mask(inv_mask, image, depth)
    # cv2.imshow('image', inv_mask_image)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    c_mask = dt.circle_mask(mask)
    xyz = [x,y,depth[x,y]]
    c_mask_image, c_mask_depth  = dt.depth_mask(c_mask, image, depth)


    # --POINT COULD--
    #Object
    o_pcl = pc.creation(mask_image, mask_depth)
    #Fingers
    f_pcl = 2#pc.creation(inv_mask_image, inv_mask_depth)
    #Combined
    c_pcl = 3#pc.creation(c_mask_image, c_mask_depth)

   


    return mask_image, inv_mask_image, c_mask_image, o_pcl, f_pcl, c_pcl

def MatplotlibClearMemory():
    #usedbackend = matplotlib.get_backend()
    #matplotlib.use('Cairo')
    allfignums = plt.get_fignums()
    for i in allfignums:
        fig = plt.figure(i)
        fig.clear()
        matplotlib.pyplot.close( fig )
    #matplotlib.use(usedbackend) 


item_lst = ['coke', 'coffee', 'chocolate', 'chips', 'apple', 'crackers', 'strawberry', 'mustard','banana','pear']
item_num = [128, 112, 128, 96, 128, 128, 128, 128, 128, 128, 128]

item_lst = ['chocolate', 'chips', 'apple', 'crackers', 'strawberry', 'mustard','banana','pear']
item_num = [128, 96, 128, 128, 128, 128, 128, 128, 128]

item_lst = ['coke']
item_num = [16]

for index in range(len(item_lst)):
    item = item_lst[index]
    for i in range(item_num[index]):
        if item == 'chocolate':
            if i < 116:
                continue
            else:
                id = i
        else:
            id = i
        print(f'Running Detection: {item} - img{id}')
        path = "D:/School/UofT/Research/images/test/"
        item_test = item + '_test'
        image = cv2.imread(path + item_test + f'/images/img{id}.png')
        depth = np.load(path + item_test + f'/depth_images/img{id}.npy')
        o_image, f_image, c_image, o_pcl, f_pcl, c_pcl = runCode(image, depth)
        #np.save(f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/valid_dataset/c_vox/grid{num}.npy', c_vox)
        o_out = path + item_test + f'/o_pcl/img{id}.ply'
        f_out = path + item_test + f'/f_pcl/img{id}.ply'
        c_out = path + item_test + f'/c_pcl/img{id}.ply'
        o_img_out = path + item + f'/rgb/o_rgb/img{id}.png'
        f_img_out = path + item + f'/rgb/f_rgb/img{id}.png'
        c_img_out = path + item + f'/rgb/c_rgb/img{id}.png'

        print("Saving Image")
        # cv2.imwrite(o_img_out, o_image)
        # cv2.imwrite(f_img_out, f_image)
        # cv2.imwrite(c_img_out, c_image)
        
        
        #Get system process information for printing memory usage:
        process = psutil.Process(os.getpid())
        
        #Clear the all the figures and check memory usage:
        MatplotlibClearMemory( )
        print('AfterDeletion: ', process.memory_info().rss)  # in bytes

        print("Saving Point Clouds\n")
        # o3d.io.write_point_cloud(o_out, o_pcl)  
        # o3d.io.write_point_cloud(f_out, f_pcl)  
        # o3d.io.write_point_cloud(c_out, c_pcl)  


        

    


