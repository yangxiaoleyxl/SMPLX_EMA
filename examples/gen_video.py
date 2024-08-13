import os
from moviepy.editor import ImageSequenceClip

# Define the image directory and output video file path
image_dir = "/Users/lxy/Gitlab/smplx/results/demo00365/images"
output_video_path = "/Users/lxy/Gitlab/smplx/results/demo00365/output_0.8_0.1.mp4"

# Get a sorted list of image file paths
image_files = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".png")])

# Create a video clip from the images
clip = ImageSequenceClip(image_files, fps=24)  # Adjust fps (frames per second) as needed

# Write the video file
clip.write_videofile(output_video_path, codec="libx264")

print(f"Video saved at {output_video_path}")
