# import smplx
# import torch
# import numpy as np
# import torch
# from einops import rearrange

# model = smplx.create("./smplx/SMPLX_FEMALE.npz", model_type="smplx")
# # data = np.load('./data/00001_0.npz') 
# data = np.load('/Users/lxy/Gitlab/smplx/results/demo00365/smplx_ema_splits_merged/time_step_017.npz') 
# betas, expression, body_pose = data['betas'], data['expression'], rearrange(data['body_pose'], 'k c -> 1 k c')
# global_orient, transl = data['global_orient'], data['transl']
# global_orient[0, 0] += np.pi
# print(betas.shape, expression.shape, body_pose.shape, global_orient.shape, transl.shape)
# betas, expression, body_pose = torch.from_numpy(betas), torch.from_numpy(expression), torch.from_numpy(body_pose)
# global_orient, transl = torch.from_numpy(global_orient), torch.from_numpy(transl)
# output = model(betas=betas, expression=expression, body_pose=body_pose, global_orient=global_orient, transl=transl, return_verts=True)

# vertices = output.vertices.detach().cpu().numpy().squeeze()
# joints = output.joints.detach().cpu().numpy().squeeze()

# import open3d as o3d 

# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# mesh.triangles = o3d.utility.Vector3iVector(model.faces)

# # o3d.visualization.draw_geometries([mesh])

# # Set up the visualization window
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(mesh)

# # Update the renderer to reflect any changes
# vis.update_geometry(mesh)
# vis.poll_events()
# vis.update_renderer()

# # Capture and save the image
# vis.capture_screen_image("mesh_output1.png", do_render=True)

# Destroy the visualizer window
# vis.destroy_window() 


# Loop for saving pictures 
import smplx
import torch
import numpy as np
import os
from einops import rearrange
import open3d as o3d 

base_path = '/Users/lxy/Gitlab/smplx/results' 
video_num = 'demo00009' 

# Set up the SMPL-X model
model = smplx.create("./smplx/SMPLX_FEMALE.npz", model_type="smplx")

# Define the data directory and image output directory
# data_dir = f"{base_path}/{video_num}/smplx_ema_splits_merged" 
data_dir = f"{base_path}/{video_num}/smplx"  
output_dir = f"{base_path}/{video_num}/images"   

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over each npz file in the directory
for file_name in sorted(os.listdir(data_dir))[:120]:
    if file_name.endswith('.npz'):
        # Load the data
        data_path = os.path.join(data_dir, file_name)
        data = np.load(data_path)
        
        # Extract the required parameters
        betas, expression, body_pose = data['betas'], data['expression'], rearrange(data['body_pose'], 'k c -> 1 k c')
        global_orient, transl = data['global_orient'], data['transl']
        global_orient[0, 0] += np.pi
        
        # Convert numpy arrays to torch tensors
        betas, expression, body_pose = torch.from_numpy(betas), torch.from_numpy(expression), torch.from_numpy(body_pose)
        global_orient, transl = torch.from_numpy(global_orient), torch.from_numpy(transl)
        
        # Generate the SMPL-X model output
        output = model(betas=betas, expression=expression, body_pose=body_pose, global_orient=global_orient, transl=transl, return_verts=True)
        
        # Extract vertices and joints
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        joints = output.joints.detach().cpu().numpy().squeeze()

        # Create an Open3D mesh from the vertices and faces
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(model.faces)
        
        # Set up the visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(mesh)

        # Update the renderer to reflect any changes
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        
        # Create output image file path
        output_image_path = os.path.join(output_dir, f"mesh_{file_name.replace('.npz', '.png')}")
        
        # Capture and save the image
        vis.capture_screen_image(output_image_path, do_render=True)
        
        # Destroy the visualizer window
        vis.destroy_window()
        
        print(f"Saved: {output_image_path}")
