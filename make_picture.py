import os
from PIL import Image

def create_image_grid(image_folder, output_file, N, M, img_width=32, img_height=32):
    image_files = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]

    total_images_needed = N * M
    if len(image_files) < total_images_needed:
        raise ValueError(f"Not enough images in the folder. Need {total_images_needed}, but found {len(image_files)}.")

    grid_img = Image.new('RGB', (M * img_width, N * img_height))
    for i in range(N):
        for j in range(M):
            img_idx = i * M + j
            img_path = image_files[img_idx]
            img = Image.open(img_path).resize((img_width, img_height))
            grid_img.paste(img, (j * img_width, i * img_height))

    grid_img.save(output_file)
create_image_grid('emasamples/0/0/', 'output_image_grid.png', N=8, M=12, img_width=32, img_height=32)
