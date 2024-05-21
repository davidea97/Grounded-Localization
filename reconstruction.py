import numpy as np
import open3d as o3d
import cv2

def depth_to_point_cloud(depth, intrinsic, extrinsic=np.eye(4)):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    z = depth
    x = (i - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (j - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    points = np.hstack((points, np.ones((points.shape[0], 1))))  # Convert to homogeneous coordinates
    points = points @ extrinsic.T
    return points[:, :3]  # Convert back to 3D

def merge_point_clouds(point_clouds):
    merged_points = np.vstack(point_clouds)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    return pcd

def main():
    # Load images, depth maps, and camera poses
    img1 = cv2.imread('data/color/0049.png')
    depth1 = cv2.imread('data/depth/0049.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    img2 = cv2.imread('data/color/0063.png')
    depth2 = cv2.imread('data/depth/0063.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    # Intrinsic camera parameters
    fx, fy = 641.2095947265625, 640.4234619140625  # Focal length
    cx, cy = 635.85791015625, 366.3246154785156  # Principal point
    intrinsic = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
    
    # Extrinsic camera parameters (4x4 transformation matrix)


    pose1 = np.array([[5.829747117431832715e-01, 7.990484137107095597e-01, -1.471803144515546058e-01, 6.189727758304701677e-01],
                      [6.506872264601017974e-01, -3.506730441877625171e-01, 6.735239478789863954e-01, -1.723982565374126974e-01],
                      [4.865660735391805214e-01, -4.884157795949945480e-01, -7.243641671601167564e-01, 4.707007342830713537e-01],
                      [0, 0, 0, 1]])  # Identity matrix for the first camera
    pose2 = np.array([[8.145450477137667900e-01, -1.024670223184950268e-01, 5.709788295449073070e-01, 2.013943256135465876e-01],
                      [-3.449900303117525713e-01, -8.768666377937733847e-01, 3.347935370224468521e-01, -3.924648747074333532e-01],
                      [4.663669899984300704e-01, -4.696864209154060776e-01, -7.495975345301306714e-01, 4.968976034412421749e-01],
                      [0, 0, 0, 1]])  # Example transformation for the second camera
    
    # Convert depth maps to point clouds
    pc1 = depth_to_point_cloud(depth1, intrinsic, pose1)
    pc2 = depth_to_point_cloud(depth2, intrinsic, pose2)
    
    # Merge point clouds
    merged_pcd = merge_point_clouds([pc1, pc2])
    
    # Visualize the reconstructed scene
    o3d.visualization.draw_geometries([merged_pcd])
    
    # Optionally save the merged point cloud
    o3d.io.write_point_cloud('reconstructed_scene.ply', merged_pcd)

if __name__ == '__main__':
    main()