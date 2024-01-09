import open3d as o3d
import numpy as np
import copy
import csv

DEBUG = False

# Define a function to save the voxel grid as a numpy array
def save_voxel_grid(grid, filename, target):
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
    np.save(f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/new_cnn/chips/data_aug/{target}/'+filename, voxel_grid)




def write_new_label(filename, newlabel1, newlabel2):
    file.write(f"{filename}, {newlabel1}, {newlabel2}\n")
    print(f'Writing labels: {filename}, new grasp direction:{newlabel1}, new grasp mode: {newlabel2}')
    file.flush()  # Writes to file right away

def get_new_label(n, oldlabel_n):

    oldlabel1 = int(oldlabel_n[1])
    oldlabel2 = int(oldlabel_n[2])
    print(f'Reading labels: {oldlabel_n[0]}, old grasp direction:{oldlabel1}, old grasp mode: {oldlabel2}')

    if (oldlabel1 == 1):
        newlabel1_t90 = 1
        newlabel1_t180 = 1
        newlabel1_t270 = 1
    elif (oldlabel1 == 2):
        newlabel1_t90 = 5
        newlabel1_t180 = 3
        newlabel1_t270 = 4
    elif (oldlabel1 == 3):
        newlabel1_t90 = 4
        newlabel1_t180 = 2
        newlabel1_t270 = 5
    elif (oldlabel1 == 4):
        newlabel1_t90 = 2
        newlabel1_t180 = 5
        newlabel1_t270 = 3
    elif (oldlabel1 == 5):
        newlabel1_t90 = 3
        newlabel1_t180 = 4
        newlabel1_t270 = 2
    else:
        print('something wrong about labeling!')
    
    newlabel2 = oldlabel2

    filename_t0 = f'voxel_grid{n}_t0'    
    filename_t90 = f'voxel_grid{n}_t90'
    filename_t180 = f'voxel_grid{n}_t180'
    filename_t270 = f'voxel_grid{n}_t270'

    for i in range(7):
        write_new_label(f'{filename_t0}_{i}.npy', oldlabel1, oldlabel2)
    for i in range(7):
        write_new_label(f'{filename_t90}_{i}.npy', newlabel1_t90, newlabel2)
    for i in range(7):
        write_new_label(f'{filename_t180}_{i}.npy', newlabel1_t180, newlabel2)
    for i in range(7):
        write_new_label(f'{filename_t270}_{i}.npy', newlabel1_t270, newlabel2)

def transform_smalldeg(pcd, filename, target):
    T_5 = np.eye(4)
    T_5[:3,:3] = frame.get_rotation_matrix_from_xyz((0, 0, np.pi/36))

    T_10 = np.eye(4)
    T_10[:3,:3] = frame.get_rotation_matrix_from_xyz((0, 0, np.pi/18))

    T_15 = np.eye(4)
    T_15[:3,:3] = frame.get_rotation_matrix_from_xyz((0, 0, np.pi/12))

    T__5 = np.eye(4)
    T__5[:3,:3] = frame.get_rotation_matrix_from_xyz((0, 0, -np.pi/36))

    T__10 = np.eye(4)
    T__10[:3,:3] = frame.get_rotation_matrix_from_xyz((0, 0, -np.pi/18))

    T__15 = np.eye(4)
    T__15[:3,:3] = frame.get_rotation_matrix_from_xyz((0, 0, -np.pi/12))

    pcd_t5 = copy.deepcopy(pcd).transform(T_5)
    pcd_t10 = copy.deepcopy(pcd).transform(T_10)
    pcd_t15 = copy.deepcopy(pcd).transform(T_15)
    pcd_t_5 = copy.deepcopy(pcd).transform(T__5)
    pcd_t_10 = copy.deepcopy(pcd).transform(T__10)
    pcd_t_15 = copy.deepcopy(pcd).transform(T__15)

    grid_t5   = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_t5, 1)
    grid_t10  = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_t10, 1)
    grid_t15  = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_t15, 1)
    grid_t_5  = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_t_5, 1)
    grid_t_10 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_t_10, 1)
    grid_t_15 = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_t_15, 1)

    if DEBUG == True:
        o3d.visualization.draw_geometries([grid_t5, frame])
        o3d.visualization.draw_geometries([grid_t10, frame])
        o3d.visualization.draw_geometries([grid_t15, frame])
        o3d.visualization.draw_geometries([grid_t_5, frame])
        o3d.visualization.draw_geometries([grid_t_10, frame])
        o3d.visualization.draw_geometries([grid_t_15, frame])

    filename1 = f'{filename}_1.npy'
    filename2 = f'{filename}_2.npy'
    filename3 = f'{filename}_3.npy'
    filename4 = f'{filename}_4.npy'
    filename5 = f'{filename}_5.npy'
    filename6 = f'{filename}_6.npy'

    save_voxel_grid(grid_t5  , filename1, target)
    save_voxel_grid(grid_t10 , filename2, target)
    save_voxel_grid(grid_t15 , filename3, target)
    save_voxel_grid(grid_t_5 , filename4, target)
    save_voxel_grid(grid_t_10, filename5, target)
    save_voxel_grid(grid_t_15, filename6, target)


if __name__ == "__main__":

    with open('/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/new_cnn/chips/labels.csv') as csvfile:
        labelreader = csv.reader(csvfile)
        header = []
        header = next(labelreader)
        oldlabel = []
        for row in labelreader:
            oldlabel.append(row)

    file = open('/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/new_cnn/chips/new_labels.csv', 'w+')
    file.write(f"voxel grid name, new grasp direction, new grasp mode\n")
    file.flush()  # Writes to file right away

    target_list = ['obj','fin','com']

    for target in target_list:
        for n in range(1):
            print(f'Image {n}/128')
            id = n

            pcd = o3d.io.read_point_cloud(f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/new_cnn/chips/pcd/{target}/img{id}.ply')
            pcd = pcd.scale(63 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=35, origin=[0, 0, 0])

            #Transformation
            T_90 = np.eye(4)
            T_90[:3,:3] = frame.get_rotation_matrix_from_xyz((0, 0, np.pi/2))

            T_180 = np.eye(4)
            T_180[:3,:3] = frame.get_rotation_matrix_from_xyz((0, 0, np.pi))

            T_270 = np.eye(4)
            T_270[:3,:3] = frame.get_rotation_matrix_from_xyz((0, 0, -np.pi/2))

            pcd_t0 = copy.deepcopy(pcd)
            filename_t0 = f'voxel_grid{n}_t0_0.npy'
            grid_t0= o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_t0, 1)       
            save_voxel_grid(grid_t0, filename_t0, target)
            filename_t0_new = f'voxel_grid{n}_t0'
            transform_smalldeg(pcd_t0, filename_t0_new, target)

            pcd_t90 = copy.deepcopy(pcd).transform(T_90)
            filename_t90 = f'voxel_grid{n}_t90_0.npy'
            grid_t90= o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_t90, 1)       
            save_voxel_grid(grid_t90, filename_t90, target)
            filename_t90_new = f'voxel_grid{n}_t90'
            transform_smalldeg(pcd_t90, filename_t90_new, target)

            pcd_t180 = copy.deepcopy(pcd).transform(T_180)
            filename_t180 = f'voxel_grid{n}_t180_0'
            grid_t180= o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_t180, 1)       
            save_voxel_grid(grid_t180, filename_t180, target)
            filename_t180_new = f'voxel_grid{n}_t180'
            transform_smalldeg(pcd_t180, filename_t180, target)

            pcd_t270 = copy.deepcopy(pcd).transform(T_270)
            filename_t270 = f'voxel_grid{n}_t270_0'
            grid_t270= o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_t270, 1)       
            save_voxel_grid(grid_t270, filename_t270, target)
            filename_t270_new = f'voxel_grid{n}_t270'
            transform_smalldeg(pcd_t270, filename_t270, target)

            get_new_label(n, oldlabel[n])
