import dataclasses
from typing import Optional

import cv2
import numpy as np
from cv2 import dnn_superres
from matplotlib import pyplot as plt

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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            r, g, b = cv2.split(frame_rgb)
            y = .299 * r + .587 * g + .114 * b
            cb = 128 - .168736 * r - .331364 * g + .5 * b
            cr = 128 + .5 * r - .418688 * g - .081312 * b

            SSV = 2
            SSH = 2
            crf = cv2.boxFilter(cr, ddepth=-1, ksize=(2, 2))
            cbf = cv2.boxFilter(cb, ddepth=-1, ksize=(2, 2))
            crsub = append_zeros(crf[::SSV, ::SSH])
            cbsub = append_zeros(cbf[::SSV, ::SSH])

            self.channels = Channels(
                is_encoded=False,
                luminosity=y.astype(np.uint8),
                chromaticCb=crsub.astype(np.uint8),
                chromaticCr=cbsub.astype(np.uint8),
            )
        if channels is not None:
            assert channels.is_encoded is False
            self.channels = channels

        self.is_key_frame = is_key_frame
        self.width = width
        self.height = height

    def show_frame(self):

        y = np.array(self.channels.luminosity)

        cb = np.array(cv2.resize(self.channels.chromaticCb, (self.width, self.height)))

        cr = np.array(cv2.resize(self.channels.chromaticCr, (self.width, self.height)))

        rec_frame = cv2.merge([y, cb, cr]).astype(np.uint8)

        xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        rgb = rec_frame.astype(np.float)
        rgb[:, :, [1, 2]] -= 128
        rgb = rgb.dot(xform.T)
        np.putmask(rgb, rgb > 255, 255)
        np.putmask(rgb, rgb < 0, 0)
        rec_frame = cv2.cvtColor(np.uint8(rgb), cv2.COLOR_RGB2BGR)
        cv2.imshow('res', rec_frame)
        cv2.waitKey(0)

        # cv2.imshow("dd", rec_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def upsample_image(self):

        DecAll = cv2.merge(
            [
                cv2.resize(self.channels.luminosity, (self.width, self.height)),
                cv2.resize(self.channels.chromaticCr, (self.width, self.height)),
                cv2.resize(self.channels.chromaticCb, (self.width, self.height)),

            ]
        )

        reImg = cv2.cvtColor(DecAll.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

        sr = dnn_superres.DnnSuperResImpl_create()

        path = "EDSR_x3.pb"
        sr.readModel(path)
        sr.setModel("edsr", 3)
        upsmpled_img = sr.upsample(reImg)

        return cv2.resize(upsmpled_img, (self.width, self.height))


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
