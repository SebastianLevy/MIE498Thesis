import numpy as np
import open3d as o3d
import cv2

# Intel L515 Intrinsics
fx = 1024
fy = 1024
cx = 319.5
cy = 239.5

voxel_size=0.00001
nb_neighbors=1000
std_ratio=0.001


temp = []
files = []

f = open("D:/School/UofT/Research/MaskRCNN/final_list.txt", "r")
for x in f:
    temp.append(int(x[:]))
for i in range(559):
     if temp.count(i) == 0:
          files.append(i)


def rso_filter(pcl, voxel, nb, std):
    # Load point cloud data

    #print("Downsample the point cloud with a voxel of 0.02")
    pcd = pcl.voxel_down_sample(voxel_size=voxel)
    
    # Create a statistical outlier removal filter object
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    #display_inlier_outlier(pcd, ind)

    
    # Apply the filter to the point cloud and save the result
    pcd_filtered = pcd.select_by_index(ind)
    
    #o3d.io.write_point_cloud(f'filtered_point_clouds/img{id}.ply', pcd_filtered)
    return(pcd_filtered)

def z_filter(pcd1,thresh):
    #pcd1 = o3d.io.read_point_cloud(f'point_clouds/img{id}.ply')
    points = np.asarray(pcd1.points)

    # Remove points over the plane
    z_threshold = thresh
    keep_z = points[:, 2] < z_threshold
    pcd1 = pcd1.select_by_index(np.where(keep_z)[0])

    return(pcd1)

    

def create_point_cloud(d_img, img):
    # Create Open3D point cloud from depth array and RGB image
    o3d_img = o3d.geometry.Image(img)
    o3d_depth_img = o3d.geometry.Image(d_img.astype(np.uint16))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1024, height=1024, fx=fx, fy=fy, cx=cx, cy=cy)
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    rgb_d_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_img,
                                                                    o3d_depth_img,
                                                                    convert_rgb_to_intensity=False)
    # o3d.visualization.draw_geometries([rgb_d_img])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgb_d_img,
                                                         intrinsic)
    return pcd

def creation(i,d):
    pcl = create_point_cloud(d,i)
    zfl = z_filter(pcl,1.75)
    if not o3d.geometry.PointCloud.has_points(zfl):
        zfl = z_filter(pcl,2.67)
        if not o3d.geometry.PointCloud.has_points(zfl):
            zfl = pcl
    # Filter point cloud using remove_statistical_outlier
    otd = rso_filter(zfl, voxel_size, nb_neighbors,std_ratio)
    if not o3d.geometry.PointCloud.has_points(otd):
        otd = rso_filter(zfl, voxel_size, nb_neighbors,0.01)
        if not o3d.geometry.PointCloud.has_points(otd):
            otd = zfl
    return otd




for n in range(0):
    id = files[n]
    print(f'Image {files[n]}: {n}/{len(files)-1}')

    print('Object')
    # Load depth data from .npy file
    depth_data = np.load(f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/grasping_cnn/dataset/m_depth/img{id}.npy')
    depth_data = depth_data
    # Load image data from .png file
    image_data = cv2.imread(f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/grasping_cnn/dataset/m_rgb/img{id}.png')
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    # Define output file name
    output_file_name = (f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/grasping_cnn/dataset/m_pcl/img{n}.ply')
    # Create point cloud
    pcl = create_point_cloud(depth_data,image_data)
    # Filter z values
    zfl = z_filter(pcl,3)
    if not o3d.geometry.PointCloud.has_points(zfl):
        zfl = z_filter(pcl,4.5)
        if not o3d.geometry.PointCloud.has_points(zfl):
            zfl = pcl
            with open("filter_failures.txt", "a") as file:
                file.write(f'Image {files[n]}: {n}/{len(files)-1} - Object - Z Filter' + "\n")
    # Filter point cloud using remove_statistical_outlier
    otd = rso_filter(zfl, voxel_size, nb_neighbors,std_ratio)
    if not o3d.geometry.PointCloud.has_points(otd):
        otd = rso_filter(zfl, voxel_size, nb_neighbors,0.01)
        if not o3d.geometry.PointCloud.has_points(otd):
            otd = zfl
            with open("filter_failures.txt", "a") as file:
                file.write(f'Image {files[n]}: {n}/{len(files)-1} - Object - RSO Filter' + "\n")

    # Write point cloud to file
    o3d.io.write_point_cloud(output_file_name, otd)
    
    print('Fingers')
    # Load depth data from .npy file
    inv_depth_data = np.load(f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/grasping_cnn/dataset/inv_depth/img{id}.npy')
    inv_depth_data = inv_depth_data
    # Load image data from .png file
    inv_image_data = cv2.imread(f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/grasping_cnn/dataset/inv_rgb/img{id}.png')
    inv_image_data = cv2.cvtColor(inv_image_data, cv2.COLOR_BGR2RGB)
    # Define output file name
    inv_output_file_name = (f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/grasping_cnn/dataset/inv_pcl/img{n}.ply')
    # Create inverse point cloud
    inv_pcl = create_point_cloud(inv_depth_data,inv_image_data)

    # Filter inverse point cloud z values
    inv_zfl = z_filter(inv_pcl,3)
    if not o3d.geometry.PointCloud.has_points(inv_zfl):
        inv_zfl = z_filter(inv_pcl,4.5)
        if not o3d.geometry.PointCloud.has_points(inv_zfl):
            inv_zfl = inv_pcl
            with open("filter_failures.txt", "a") as file:
                file.write(f'Image {files[n]}: {n}/{len(files)-1} - Fingers - Z Filter' + "\n")

    # Filter inverse point cloud using remove_statistical_outlier
    inv_otd = rso_filter(inv_zfl, voxel_size, nb_neighbors,std_ratio)
    if not o3d.geometry.PointCloud.has_points(inv_otd):
        inv_otd = rso_filter(inv_zfl, voxel_size, nb_neighbors,0.01)
        if not o3d.geometry.PointCloud.has_points(inv_otd):
            inv_otd = inv_zfl
            with open("filter_failures.txt", "a") as file:
                file.write(f'Image {files[n]}: {n}/{len(files)-1} - Fingers - RSO Filter' + "\n")
    # Write inverse point cloud to file
    o3d.io.write_point_cloud(inv_output_file_name, inv_otd)  

    print('Combined')
    # Load depth data from .npy file
    c_depth_data = np.load(f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/grasping_cnn/dataset/c_depth/img{id}.npy')
    # Load image data from .png file
    c_image_data = cv2.imread(f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/grasping_cnn/dataset/c_rgb/img{id}.png')
    c_image_data = cv2.cvtColor(c_image_data, cv2.COLOR_BGR2RGB)
    # Define output file name
    c_output_file_name = (f'/Users/sebastian.levy42/Documents/Sebastian/School/Winter 2T3/Thesis/Code/images/grasping_cnn/dataset/c_pcl/img{n}.ply')
    # Create combined point cloud
    c_pcl = create_point_cloud(c_depth_data,c_image_data)

    # Filter combined point cloud z values
    c_zfl = z_filter(c_pcl,3)
    if not o3d.geometry.PointCloud.has_points(c_zfl):
        c_zfl = z_filter(c_pcl,4.5)
        if not o3d.geometry.PointCloud.has_points(c_zfl):
            c_zfl = c_pcl
            with open("filter_failures.txt", "a") as file:
                file.write(f'Image {files[n]}: {n}/{len(files)-1} - Fingers - Z Filter' + "\n")

    # Filter combined point cloud using remove_statistical_outlier
    c_otd = rso_filter(c_zfl, voxel_size, nb_neighbors,std_ratio)
    if not o3d.geometry.PointCloud.has_points(c_otd):
        c_otd = rso_filter(c_zfl, voxel_size, nb_neighbors,0.01)
        if not o3d.geometry.PointCloud.has_points(c_otd):
            c_otd = c_zfl
            with open("filter_failures.txt", "a") as file:
                file.write(f'Image {files[n]}: {n}/{len(files)-1} - Fingers - RSO Filter' + "\n")
    # Write combined point cloud to file
    o3d.io.write_point_cloud(c_output_file_name, c_otd)  


