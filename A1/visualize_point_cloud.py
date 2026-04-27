import open3d as o3d
pcd = o3d.io.read_point_cloud("output_cloud.ply")
o3d.visualization.draw_geometries([pcd])