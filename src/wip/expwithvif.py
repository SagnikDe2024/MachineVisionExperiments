from torchmetrics.image import VisualInformationFidelity

from src.common.common_utils import acquire_image
from src.encoder_decoder.image_reconstruction_loss import MultiscalePerceptualLoss
from src.image_cleanup.image_defects import add_jpeg_artifacts


def calc_vif_score():
	vif = VisualInformationFidelity()
	vif_score = vif(img_acq, img_acq)
	print(f'vif_score = {vif_score}')


def calc_perceptual_loss():
	perceptual_loss = MultiscalePerceptualLoss()
	loss_value = perceptual_loss(img_acq, arty)
	print(f'Perceptual loss between original and distorted image: {loss_value.item()}')


if __name__ == "__main__":
	img_acq = acquire_image('data/reddit_face.jpg')
	img_acq = img_acq.unsqueeze(0)
	arty = add_jpeg_artifacts(img_acq, 50)
	# calc_vif_score()
	calc_perceptual_loss()
