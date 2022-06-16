import dataclasses
from typing import Optional

import cv2
import numpy as np
from cv2 import dnn_superres
from matplotlib import pyplot as plt

from repository import append_zeros


@dataclasses.dataclass
class Frame:
    def __init__(
        self,
        width: int,
        height: int,
        is_key_frame: bool,
        frame=None,
        channels: Optional["Channels"] = None,
        method: int = 0,
    ):

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            r, g, b = cv2.split(frame_rgb)
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
            cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

            SSV = 2
            SSH = 2
            crf = cv2.boxFilter(cr, ddepth=-1, ksize=(2, 2))
            cbf = cv2.boxFilter(cb, ddepth=-1, ksize=(2, 2))
            crsub = append_zeros(crf[::SSV, ::SSH]) if method == 0 else cr
            cbsub = append_zeros(cbf[::SSV, ::SSH]) if method == 0 else cb

            self.channels = Channels(
                is_encoded=False,
                luminosity=y.astype(np.uint8),
                chromaticCb=cbsub.astype(np.uint8),
                chromaticCr=crsub.astype(np.uint8),
            )
        if channels is not None:
            assert channels.is_encoded is False
            self.channels = channels

        self.is_key_frame = is_key_frame
        self.width = width
        self.height = height
        self.method = method

    def build_frame(self):

        y = self.channels.luminosity[: self.height, : self.width]

        chromatic_shape = (
            (self.height // 2, self.width // 2)
            if self.method == 0
            else (self.height, self.width)
        )

        cb = self.channels.chromaticCb[: chromatic_shape[0], : chromatic_shape[1]]
        cr = self.channels.chromaticCr[: chromatic_shape[0], : chromatic_shape[1]]

        if self.method == 0:
            cb = np.array(
                cv2.resize(cb, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            )
            cr = np.array(
                cv2.resize(cr, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            )
        else:
            cb = np.array(cb)
            cr = np.array(cr)

        rec_frame = cv2.merge([y, cb, cr]).astype(np.uint8)

        xform = np.array([[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]])
        rgb = rec_frame.astype(np.float)
        rgb[:, :, [1, 2]] -= 128
        rgb = rgb.dot(xform.T)
        np.putmask(rgb, rgb > 255, 255)
        np.putmask(rgb, rgb < 0, 0)
        return cv2.cvtColor(np.uint8(rgb), cv2.COLOR_RGB2BGR)

    def show_frame(self):
        rec_frame = self.build_frame()
        cv2.imshow("", rec_frame)
        cv2.waitKey(0)

    def upsample_image(self):
        interpolation_methods = ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC"]
        interpolation_method = interpolation_methods[0]

        DecAll = cv2.merge(
            [
                cv2.resize(
                    self.channels.luminosity,
                    (self.width, self.height),
                    interpolation=getattr(cv2, interpolation_method),
                ),
                cv2.resize(
                    self.channels.chromaticCr,
                    (self.width, self.height),
                    interpolation=getattr(cv2, interpolation_method),
                ),
                cv2.resize(
                    self.channels.chromaticCb,
                    (self.width, self.height),
                    interpolation=getattr(cv2, interpolation_method),
                ),
            ]
            if self.method == 0
            else [
                self.channels.luminosity,
                self.channels.chromaticCr,
                self.channels.chromaticCb,
            ]
        )

        reImg = cv2.cvtColor(DecAll.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

        if self.method == 1:
            sr = dnn_superres.DnnSuperResImpl_create()
            path = "EDSR_x3.pb"
            sr.readModel(path)
            sr.setModel("edsr", 3)
            upsmpled_img = sr.upsample(reImg)
        else:
            upsmpled_img = reImg

        if self.method == 0:
            return cv2.resize(upsmpled_img, (self.width, self.height))
        else:

            mask = np.zeros((self.height, self.width, 3))
            mask[: self.height, : self.width] = upsmpled_img[
                : self.height, : self.width
            ]
            return mask.astype(np.uint8)


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
