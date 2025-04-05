from dataclasses import dataclass


@dataclass
class NewSize:
	starting_size: int
	ratio: float


@dataclass
class SquareKernel:
	kernel_size: int


@dataclass
class SeparatedKernel:
	kernel_size: int
	parameter_ratio: float
	intermediate_ratio: float


CNNKernel = SquareKernel | SeparatedKernel
