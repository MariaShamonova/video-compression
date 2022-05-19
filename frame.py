import dataclasses
from typing import Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt

from repository import round_num

def append_zeros(frame):
	width, height = frame.shape
	mask = np.zeros((round_num(width), round_num(height)))
	mask[:width, :height] = np.array(frame)
	return mask


@dataclasses.dataclass
class Frame:
	def __init__(self, is_key_frame: bool, frame=None, channels: Optional['Channels'] = None):
		if frame is not None:
			ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
			SSV = 2
			SSH = 2
			crf = cv2.boxFilter(ycbcr[:, :, 1], ddepth=-1, ksize=(2, 2))
			cbf = cv2.boxFilter(ycbcr[:, :, 2], ddepth=-1, ksize=(2, 2))
			crsub = append_zeros(crf[::SSV, ::SSH])
			cbsub = append_zeros(cbf[::SSV, ::SSH])

			self.channels = Channels(is_encoded=False, luminosity=ycbcr[:, :, 0], chromaticCb=cbsub, chromaticCr=crsub)
		if channels is not None:
			assert channels.is_encoded is False
			self.channels = channels

		self.is_key_frame = is_key_frame

	def show_frame(self):
		w, h = self.channels.luminosity.shape
		DecAll = cv2.merge([cv2.resize(self.channels.luminosity, (h, w)), cv2.resize(self.channels.chromaticCb, (h, w)), cv2.resize(self.channels.chromaticCr, (h, w))])


		reImg = cv2.cvtColor(DecAll.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

		#
		# img3 = np.zeros(self.channels.luminosity.shape, np.uint8)
		# img3[:, :, 0] = reImg[:, :, 2]
		# img3[:, :, 1] = reImg[:, :, 1]
		# img3[:, :, 2] = reImg[:, :, 0]

		# SSE = np.sqrt(np.sum((self.channels.chromaticCb - img3) ** 2))
		# print
		# "Sum of squared error: ", SSE
		cv2.imshow('dd', reImg)
		cv2.waitKey(0)  # wait for a keyboard input
		cv2.destroyAllWindows()



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

