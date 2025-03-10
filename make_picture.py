import os
from PIL import Image
import math

def get_image_files_recursively(folder):
    """Recursively collect all image files from a folder and its subfolders."""
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                image_files.append(os.path.join(root, file))
    return image_files

def create_image_grids(image_folder, output_prefix, N, M, img_width=32, img_height=32):
    # Use recursive function to get all image files
    image_files = get_image_files_recursively(image_folder)
    
    if not image_files:
        raise ValueError(f"No valid images found in the folder {image_folder} or its subfolders.")
    
    # Calculate how many grids we need
    total_images = len(image_files)
    images_per_grid = N * M
    num_grids = math.ceil(total_images / images_per_grid)
    
    for grid_num in range(num_grids):
        # Calculate start and end indices for this grid
        start_idx = grid_num * images_per_grid
        end_idx = min(start_idx + images_per_grid, total_images)
        current_batch = image_files[start_idx:end_idx]
        
        # Create a grid (potentially partially filled)
        grid_img = Image.new('RGB', (M * img_width, N * img_height))
        
        for idx, img_path in enumerate(current_batch):
            i = idx // M  # Row
            j = idx % M   # Column
            
            img = Image.open(img_path).resize((img_width, img_height))
            grid_img.paste(img, (j * img_width, i * img_height))
        
        # Save this grid
        output_file = f"{output_prefix}_{grid_num+1}.png"
        grid_img.save(output_file)
        print(f"Created grid {grid_num+1}/{num_grids} with {len(current_batch)} images")

# Create grids with all images in the 'out' folder
create_image_grids('samples_280/0', 'output_image_grid', N=8, M=24, img_width=32, img_height=32)
