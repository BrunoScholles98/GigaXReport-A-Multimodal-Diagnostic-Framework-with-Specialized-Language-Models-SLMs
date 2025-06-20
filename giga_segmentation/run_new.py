import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from DC_UNet import DC_Unet

# Define device for computation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NAME_CHECKPOINT = 'checkpoint.pth'
IN_CHANNELS = 1

class AteromaTestDataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root)
                       if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image_path = self.images[self.index]
        image = Image.open(image_path).convert('L')  # Assuming grayscale images
        original_size = image.size  # (width, height)
        name = os.path.basename(image_path)
        if name.endswith('.jpg'):
            name = name.replace('.jpg', '.png')
        self.index += 1
        return image, name, original_size  # Return the original image and its size

def validate(model, loader, save_test_images=False):
    model.eval()
    loader.index = 0
    loader_size = loader.size

    for _ in range(loader_size):
        # Load data
        original_image, name, original_size = loader.load_data()
        width, height = original_size  # Note: PIL size is (width, height)

        # Divide the image into 2 rows and 3 columns (6 cells)
        num_rows = 2
        num_cols = 3
        cell_width = width // num_cols
        cell_height = height // num_rows

        # Create an empty mask for the entire image
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # Define the positions of the cells we need to process
        # Bottom-left cell: last row, first column (row 1, col 0)
        # Bottom-right cell: last row, last column (row 1, col 2)

        # Cell positions are zero-based indices
        cells_to_process = [(1, 0), (1, 2)]  # (row_index, col_index)

        for row_idx, col_idx in cells_to_process:
            # Calculate the coordinates of the cell
            left = col_idx * cell_width
            upper = row_idx * cell_height
            # For last column/row, make sure to include the remaining pixels
            right = (col_idx + 1) * cell_width if col_idx < num_cols -1 else width
            lower = (row_idx + 1) * cell_height if row_idx < num_rows -1 else height

            # Extract the cell image
            cell_image = original_image.crop((left, upper, right, lower))

            # Preprocess the cell image
            cell_image_resized = cell_image.resize((loader.testsize, loader.testsize))
            cell_image_tensor = transforms.ToTensor()(cell_image_resized)
            cell_image_tensor = transforms.Normalize([0.5], [0.5])(cell_image_tensor)
            cell_image_tensor = cell_image_tensor.unsqueeze(0).to(DEVICE)

            # Apply the model
            with torch.no_grad():
                prediction = model(cell_image_tensor)

            # Process prediction
            prediction = prediction.sigmoid().cpu().numpy().squeeze()
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            prediction = (prediction >= 0.5).astype(np.uint8)

            # Resize prediction back to the cell size
            prediction_resized = Image.fromarray(prediction * 255).resize((right - left, lower - upper), resample=Image.NEAREST)
            prediction_resized_np = np.array(prediction_resized) // 255  # Convert back to binary mask (0 or 1)

            # Place the prediction in the combined mask
            combined_mask[upper:lower, left:right] = prediction_resized_np

        # Now, save the combined mask and overlay image
        if save_test_images:
            # Save the combined prediction mask
            output_dir = 'predictions'  # Specify the output directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save combined mask
            combined_mask_image = Image.fromarray(combined_mask * 255)
            combined_mask_image.save(os.path.join(output_dir, name))

            # Create overlay image
            # Convert original image to RGBA
            original_image_rgba = original_image.convert('RGBA')

            # Create red overlay where mask is present
            red_overlay = np.zeros((height, width, 4), dtype=np.uint8)
            red_overlay[..., 0] = 255  # Red channel
            red_overlay[..., 3] = (combined_mask > 0) * 128  # Alpha channel, 128 for 50% transparency

            # Convert red_overlay to Image
            red_overlay_image = Image.fromarray(red_overlay, mode='RGBA')

            # Overlay the red mask on the original image
            overlay_image = Image.alpha_composite(original_image_rgba, red_overlay_image)

            # Save the overlay image
            overlay_image.save(os.path.join(output_dir, 'overlay_' + name))

def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

if __name__ == '__main__':
    # Initialize the model
    model = DC_Unet(IN_CHANNELS).to(DEVICE)
    # Load the checkpoint
    load_checkpoint(NAME_CHECKPOINT, model)
    # Define the image root and test size
    image_root = r"E:\Igor\Downloads\det\giga_new\images"  # Replace with your images directory
    testsize = 352  # Or the size your model expects
    # Create the dataset loader
    loader = AteromaTestDataset(image_root=image_root, testsize=testsize)
    # Run validation/inference
    validate(model, loader, save_test_images=True)
