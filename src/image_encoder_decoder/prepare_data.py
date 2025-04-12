import torch
class PrepareData:
	def __init__(self):
		raw_data_location = 'C:\mywork\ml\MachineVisionExperiments\data\raw'
		prepared_data_location = 'C:\mywork\ml\MachineVisionExperiments\data\prepared'

	def prepare_image(self, image_path):
		image_name = torch.load(image_path,mode= )


