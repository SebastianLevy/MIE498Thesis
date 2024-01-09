import open3d as o3d
import numpy as np


#cd = o3d.io.read_point_cloud(f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/MaskRCNN/main/Mask_RCNN/images/grasping_cnn/dataset/inv_fpc/img21.ply')

def save_voxel_grid(grid):
    # Get the indices of the voxels in the grid
    voxels_ind = np.array([ind.grid_index for ind in grid.get_voxels()])
    # Get the RGB color of each voxel in the grid
    voxel_rgb = np.array([ind.color*15 for ind in grid.get_voxels()])

    # Create a numpy array to store the voxel grid
    voxel_grid = np.zeros((64, 64, 64, 3), dtype=np.uint8)
    # Loop through each voxel in the grid
    for i, voxel in enumerate(voxels_ind):
        # Check if the voxel is outside the 64x64x64 grid
        if (voxel[0]>=64) or (voxel[1]>=64) or (voxel[2]>=64):
            continue  # If so, skip it
        # Otherwise, set the RGB values of the voxel in the voxel grid
        voxel_grid[voxel[0], voxel[1], voxel[2], 0] = int(voxel_rgb[i][0])
        voxel_grid[voxel[0], voxel[1], voxel[2], 1] = int(voxel_rgb[i][1])
        voxel_grid[voxel[0], voxel[1], voxel[2], 2] = int(voxel_rgb[i][2])

    # Save the voxel grid as a numpy file
    #np.save('/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/grasping_cnn/data_aug/fin/'+filename, voxel_grid)
    return voxel_grid

def pcl_to_voxel(pcd):
    pcd = pcd.scale(63 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
    grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 1)
    vox_grid = save_voxel_grid(grid)       
    return grid, vox_grid

def concat(obj,fin,com):
    concat = np.concatenate((obj,fin,com), axis=0)
    return concat


