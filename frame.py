import dataclasses
from typing import Optional

import cv2
import numpy as np
from cv2 import dnn_superres
from repository import round_num


def append_zeros(frame):
    width, height = frame.shape
    mask = np.zeros((round_num(width), round_num(height)))
    mask = mask.astype("uint8")
    mask[:width, :height] = np.array(frame)
    return mask


@dataclasses.dataclass
class Frame:
    def __init__(
        self,
        width: int,
        height: int,
        is_key_frame: bool,
        frame=None,
        channels: Optional["Channels"] = None,
    ):

        if frame is not None:
            ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
            # cv2.imshow('ee', ycbcr)
            # cv2.waitKey(0)
            # reImg = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
            # cv2.imshow('reImg', reImg)
            # cv2.waitKey(0)
            SSV = 2
            SSH = 2
            crf = cv2.boxFilter(ycbcr[:, :, 1], ddepth=-1, ksize=(2, 2))
            cbf = cv2.boxFilter(ycbcr[:, :, 2], ddepth=-1, ksize=(2, 2))
            crsub = append_zeros(crf[::SSV, ::SSH])
            cbsub = append_zeros(cbf[::SSV, ::SSH])

            self.channels = Channels(
                is_encoded=False,
                luminosity=ycbcr[:, :, 0],
                chromaticCb=cbsub,
                chromaticCr=crsub,
            )
        if channels is not None:
            assert channels.is_encoded is False
            self.channels = channels

        self.is_key_frame = is_key_frame
        self.width = width
        self.height = height

    def show_luminosity_channel(self):
        lum = self.channels.luminosity.astype("uint8")
        cv2.imshow("show_luminosity_channel", lum)
        cv2.waitKey(0)

    def show_frame(self):

        DecAll = cv2.merge(
            [
                cv2.resize(self.channels.luminosity, (self.height, self.width)),
                cv2.resize(self.channels.chromaticCr, (self.height, self.width)),
                cv2.resize(self.channels.chromaticCb, (self.height,self.width)),

            ]
        )

        reImg = cv2.cvtColor(DecAll.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

        cv2.imshow("dd", reImg)
        cv2.waitKey(0)  # wait for a keyboard input
        cv2.destroyAllWindows()

    def upsample_image(self):

        DecAll = cv2.merge(
            [
                cv2.resize(self.channels.luminosity, (self.height, self.width)),
                cv2.resize(self.channels.chromaticCr, (self.height, self.width)),
                cv2.resize(self.channels.chromaticCb, (self.height, self.width)),

            ]
        )

        reImg = cv2.cvtColor(DecAll.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

        sr = dnn_superres.DnnSuperResImpl_create()

        path = "EDSR_x3.pb"
        sr.readModel(path)
        sr.setModel("edsr", 3)
        upsmpled_img = sr.upsample(reImg)

        return cv2.resize(upsmpled_img, (self.height, self.width))


@dataclasses.dataclass
class EncodedFrame:
    encoded_channels: "Channels" = dataclasses.field(default_factory=list)
    residual_frame: list = dataclasses.field(default_factory=list)
    motion_vectors: "Channels" = dataclasses.field(default_factory=list)
    predict_image: "Channels" = dataclasses.field(default_factory=list)


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
        return [self.luminosity, self.chromaticCr, self.chromaticCb]

    @property
    def list_motion_vectors(self) -> list:
        return [self.mv_luminosity, self.mv_chromaticCr, self.mv_chromaticCb]


@dataclasses.dataclass
class ResidualFrame:
    luminosity: list = dataclasses.field(default_factory=list)
    chromaticCb: list = dataclasses.field(default_factory=list)
    chromaticCr: list = dataclasses.field(default_factory=list)
