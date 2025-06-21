from src.common.common_utils import acquire_image
from src.image_cleanup.image_defects import add_jpeg_artifacts
from torchmetrics.image import VisualInformationFidelity

if __name__ == "__main__":
	img_acq = acquire_image('data/normal_pic.jpg')
	img_acq = img_acq.unsqueeze(0)
	arty = add_jpeg_artifacts(img_acq,100)
	vif = VisualInformationFidelity()
	vif_score = vif(img_acq, img_acq)

	print(f'vif_score = {vif_score}')
	print(f'diff = {VisualInformationFidelity.is_differentiable}')



