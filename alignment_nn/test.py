import os
import open3d as o3d

def obj_to_pcd(input_obj_file, output_pcd_file, num_points=1000):
    # Read the mesh from the OBJ file
    mesh = o3d.io.read_triangle_mesh(input_obj_file)

    # Skip the mesh if it doesn't contain any triangles
    if len(mesh.triangles) == 0:
        print(f"Skipping {input_obj_file} as it contains no triangles.")
        return

    # Sample points from the mesh surface
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    # Save the point cloud to a PCD file
    o3d.io.write_point_cloud(output_pcd_file, point_cloud)

def main():
    # Directory containing the OBJ files
    input_directory = '../data'
    
    # Iterate through all the OBJ files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.obj'):
            input_obj_path = os.path.join(input_directory, filename)
            
            # Generate output PCD filename
            output_pcd_filename = os.path.splitext(filename)[0] + '.pcd'
            output_pcd_path = os.path.join(input_directory, output_pcd_filename)
            
            # Convert OBJ to PCD
            obj_to_pcd(input_obj_path, output_pcd_path)

if __name__ == "__main__":
    main()
