from src.encoder_decoder.image_reconstruction_loss import MultiScaleGradientLoss, MultiscalePerceptualLoss, \
	ReconstructionLoss


class TrainEncoderAndDecoder:
	def __init__(self,epochs):

		self.reconstruction_loss = ReconstructionLoss()
		self.gradient_loss = MultiScaleGradientLoss()
		self.perceptual_loss = MultiscalePerceptualLoss()
		self.epochs = epochs


	def train(self, images):
		for epoch in range(1, self.epochs + 1):








