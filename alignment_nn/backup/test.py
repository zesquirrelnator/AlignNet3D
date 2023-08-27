import open3d as o3d

# Path to the input OBJ file
input_obj_file = '../data/test3/target.obj'

# Read the mesh from the OBJ file
mesh = o3d.io.read_triangle_mesh(input_obj_file)

# Sample points from the mesh surface
num_points = 1000
point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

# Save the point cloud to a PCD file
output_pcd_file = '../data/test3/source.pcd'
o3d.io.write_point_cloud(output_pcd_file, point_cloud)

# Optionally, visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
