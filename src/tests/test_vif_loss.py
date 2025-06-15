import torch
from src.common.common_utils import acquire_image
from src.image_cleanup.image_defects import add_jpeg_artifacts
from src.encoder_decoder.image_reconstruction_loss import VisualInformationFidelityLoss

if __name__ == "__main__":
    # Load an image
    img_acq = acquire_image('data/normal_pic.jpg')
    img_acq = img_acq.unsqueeze(0)
    
    # Create a distorted version of the image
    distorted_img = add_jpeg_artifacts(img_acq, 50)
    
    # Initialize the VIF loss function
    vif_loss = VisualInformationFidelityLoss()
    
    # Calculate the loss between the original and distorted images
    loss_value = vif_loss(img_acq, distorted_img)
    
    print(f"VIF Loss between original and distorted image: {loss_value.item()}")
    
    # Calculate the loss between the original and itself (should be close to 0)
    self_loss = vif_loss(img_acq, img_acq)
    
    print(f"VIF Loss between original and itself: {self_loss.item()}")