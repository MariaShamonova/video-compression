import dataclasses
from typing import Optional

import cv2
import numpy as np


@dataclasses.dataclass
class Frame:
	def __init__(self, is_key_frame: bool, frame=None, channels: Optional['Channels'] = None):
		if frame is not None:
			ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
			SSV = 2
			SSH = 2
			crf = cv2.boxFilter(ycbcr[:, :, 1], ddepth=-1, ksize=(2, 2))
			cbf = cv2.boxFilter(ycbcr[:, :, 2], ddepth=-1, ksize=(2, 2))
			crsub = crf[::SSV, ::SSH]
			cbsub = cbf[::SSV, ::SSH]
			self.channels = Channels(is_encoded=False, luminosity=ycbcr[:, :, 0], chromaticCb=cbsub, chromaticCr=crsub)
		if channels is not None:
			assert channels.is_encoded is True
			self.channels = channels

		self.is_key_frame = is_key_frame

	def show_frame(self):
		return cv2.merge([self.channels.luminosity, self.channels.chromaticCb, self.channels.chromaticCr])




@dataclasses.dataclass
class EncodedFrame:
	encoded_channels: 'Channels' = dataclasses.field(default_factory=list)
	residual_frame: list = dataclasses.field(default_factory=list)
	motion_vectors: 'Channels' = dataclasses.field(default_factory=list)
	predict_image: 'Channels'= dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Channels:
	is_encoded: bool
	luminosity: np.ndarray = dataclasses.field(default_factory=list)
	chromaticCb: np.ndarray = dataclasses.field(default_factory=list)
	chromaticCr: np.ndarray = dataclasses.field(default_factory=list)

	mv_luminosity: np.ndarray = dataclasses.field(default_factory=list)
	mv_chromaticCb: np.ndarray = dataclasses.field(default_factory=list)
	mv_chromaticCr: np.ndarray = dataclasses.field(default_factory=list)

	@property
	def list_channels(self) -> list:
		return [self.luminosity, self.chromaticCb, self.chromaticCr]

	@property
	def list_motion_vectors(self) -> list:
		return [self.mv_luminosity, self.mv_chromaticCb, self.mv_chromaticCr]



@dataclasses.dataclass
class ResidualFrame:
	luminosity: list = dataclasses.field(default_factory=list)
	chromaticCb: list = dataclasses.field(default_factory=list)
	chromaticCr: list = dataclasses.field(default_factory=list)

