import dataclasses
import math

from scipy.fft import dct
import cv2
import numpy as np
from matplotlib import pyplot as plt

from constants import (
    MATRIX_QUANTIZATION,
    MATRIX_QUANTIZATION_CHROMATIC,
    BLOCK_SIZE,
    SEARCH_AREA,
)
from frame import EncodedFrame, Frame, Channels
from repository import get_probabilities, HuffmanTree, reshape_frame, concat_blocks

from scipy.fftpack import dct, idct


@dataclasses.dataclass
class Encoder:
    def encode_traditional_method(self, frame: Frame, is_key_frame: bool) -> str:

        all_dct_elements = []
        quantized_channels = []
        if not is_key_frame:
            print("ffd")

        for idx, (channel, motion_vectors) in enumerate(
            zip(frame.channels.list_channels, frame.channels.list_motion_vectors)
        ):

            X = reshape_frame(channel, BLOCK_SIZE)
            row, column = X.shape[0], X.shape[1]

            Y = [[[] for c in range(column)] for r in range(row)]

            for i in range(0, row):
                for j in range(0, column):

                    dct_coeff = self.dct(X[i][j])

                    quantinization_coeff = self.quantization(
                        dct_coeff,
                        MATRIX_QUANTIZATION_CHROMATIC
                        if idx > 0
                        else MATRIX_QUANTIZATION,
                    )

                    sequence_coeff = self.zig_zag_transform(quantinization_coeff)

                    series_value_coeff = self.separate_pair(sequence_coeff)
                    Y[i][j] = series_value_coeff
                    if not frame.is_key_frame:
                        all_dct_elements.append(motion_vectors[i][j][0])
                        all_dct_elements.append(motion_vectors[i][j][1])

                    if len(Y[i][j]):
                        all_dct_elements.append(abs(Y[i][j][0]))
                        all_dct_elements.append(abs(Y[i][j][1]))

                        all_dct_elements.extend(
                            [
                                abs(Y[i][j][index])
                                for index in range(1, len(Y[i][j]) - 1)
                            ]
                        )

            quantized_channels.append(Y)

        frame.channels.is_encoded = True
        frame.channels.luminosity = quantized_channels[0]
        frame.channels.chromaticCr = quantized_channels[1]
        frame.channels.chromaticCb = quantized_channels[2]

        dictionary = self.entropy_encoder(all_dct_elements)
        bit_stream = self.transform_to_bit_stream(dictionary, frame)

        return bit_stream, dictionary

    def encode_nn_entropy(self, frame: Frame):

        all_dct_elements = []
        for idx, (channel, motion_vectors) in enumerate(
            zip(frame.channels.list_channels, frame.channels.list_motion_vectors)
        ):
            quantized_channels = []
            X = reshape_frame(channel, BLOCK_SIZE)
            row, column = X.shape[0], X.shape[1]
            Y = [[[] for c in range(column)] for r in range(row)]

            for i in range(0, row):
                for j in range(0, column):
                    sequence_coeff = np.array(X[i][j]).flatten("F")
                    series_value_coeff = self.separate_pair(sequence_coeff)
                    Y[i][j] = series_value_coeff

                    if len(Y[i][j]):
                        all_dct_elements.extend(
                            np.absolute(Y[i][j][: len(Y[i][j]) - 1])
                        )

            quantized_channels.append(Y)
            all_dct_elements.extend(channel)

        dictionary = self.entropy_encoder(all_dct_elements)
        bit_stream = self.transform_to_bit_stream(dictionary, frame)

        return bit_stream, dictionary

    def encode_nn(self, frame: Frame) -> Frame:
        import torch
        import torch.nn as nn
        from torchvision import transforms

        frame.channels.luminosity = transforms.ToTensor()(frame.channels.luminosity)
        frame.channels.chromaticCr = transforms.ToTensor()(frame.channels.chromaticCr)
        frame.channels.chromaticCb = transforms.ToTensor()(frame.channels.chromaticCb)

        ker = (
            torch.tensor([[0, 0, 0], [0, 0, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]])
            .type(torch.FloatTensor)
            .repeat(1, 1, 1)
        )

        conv_layer = nn.Conv2d(1, 1, 3, stride=1, bias=False)
        conv_layer.weight = torch.nn.Parameter(ker.unsqueeze(0))
        pool_layer = nn.MaxPool2d((3, 3))

        layers_pipeline = nn.Sequential(conv_layer, pool_layer)

        frame.channels.luminosity = np.array(
            layers_pipeline(frame.channels.luminosity.unsqueeze(0))
            .mul_(255.0)
            .squeeze(0)
            .squeeze(0)
            .byte()
            .cpu()
        )

        frame.channels.chromaticCb = np.array(
            layers_pipeline(frame.channels.chromaticCb.unsqueeze(0))
            .mul_(255.0)
            .squeeze(0)
            .squeeze(0)
            .byte()
            .cpu()
        )

        frame.channels.chromaticCr = np.array(
            layers_pipeline(frame.channels.chromaticCr.unsqueeze(0))
            .mul_(255.0)
            .squeeze(0)
            .squeeze(0)
            .byte()
            .cpu()
        )

        frame.channels.is_encoded = True

        return self.encode_nn_entropy(frame)

    def encode_I_frame(self, frame: Frame, method) -> Frame:

        if method == 0:
            return self.encode_traditional_method(frame=frame, is_key_frame=True)
        else:
            return self.encode_nn(frame=frame)

    def encode_B_frame(self, frame: Frame, reconstructed_frame: Frame, method) -> Frame:
        assert frame.is_key_frame is False

        residual_frame = self.motion_estimation(frame, reconstructed_frame)
        if method == 0:
            encoded_frame = self.encode_traditional_method(
                frame=residual_frame, is_key_frame=False
            )
        else:
            encoded_frame = self.encodeNN(frame=residual_frame)
        return encoded_frame

    @staticmethod
    def transform_rgb_to_ycbcr(frame):
        ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        SSV = 2
        SSH = 2
        crf = cv2.boxFilter(ycbcr[:, :, 1], ddepth=-1, ksize=(2, 2))
        cbf = cv2.boxFilter(ycbcr[:, :, 2], ddepth=-1, ksize=(2, 2))
        crsub = crf[::SSV, ::SSH]
        cbsub = cbf[::SSV, ::SSH]

        return [ycbcr[:, :, 0], crsub, cbsub]

    def dct(self, X):
        return dct(dct(X, axis=0, norm="ortho"), axis=1, norm="ortho")

    def dct_output(self, block):
        plt.figure()
        plt.imshow(
            block,
            cmap="gray",
            interpolation="nearest",
            vmax=np.max(block) * 0.01,
            vmin=0,
        )
        plt.colorbar(shrink=0.5)
        plt.title("8x8 DCT")
        plt.show()

    def quantization(self, Y, coefficient):
        quantization_coeff = np.round((np.divide(np.array(Y), coefficient)).astype(int))
        return quantization_coeff

    @staticmethod
    def zig_zag_transform(block):
        zigzag = []

        for index in range(1, len(block) + 1):
            slice = [i[:index] for i in block[:index]]

            diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]

            if len(diag) % 2:
                diag.reverse()
            zigzag += diag
        for index in reversed(range(1, len(block))):
            slice = [i[:index] for i in block[:index]]

            diag = [
                block[len(block) - index + i][len(block) - i - 1]
                for i in range(len(slice))
            ]

            if len(diag) % 2:
                diag.reverse()
            zigzag += diag

        return zigzag

    @staticmethod
    def separate_pair(seq):
        transform_seq = []
        i = len(seq) - 1

        try:
            while seq[i] == 0:
                if i == 0:
                    break
                i -= 1
        except:
            print(seq)

        count_zero = 0
        if i == 0 and seq[i] == 0:
            transform_seq = [63, 0, "EOB"]
        else:
            seq = seq[: i + 1]

            for i in range(0, len(seq)):

                if seq[i] != 0:
                    is_latest_element = "??????" if i == (len(seq) - 1) else 0
                    transform_seq.extend([count_zero, seq[i], is_latest_element])

                    count_zero = 0
                else:
                    count_zero += 1

        return transform_seq

    @staticmethod
    def entropy_encoder(blocks):

        probability = get_probabilities(blocks)
        tree = HuffmanTree(probability)
        codewars = tree.get_code()

        return codewars

    @staticmethod
    def transform_to_bit_stream(dictionary, quantized_frame: Frame) -> str:
        bit_stream_for_all_channels = ""

        for (channel, motion_vector) in zip(
            quantized_frame.channels.list_channels,
            quantized_frame.channels.list_motion_vectors,
        ):
            bit_stream = ""
            r = len(channel)
            c = len(channel[0])

            for i in range(r):
                for j in range(c):
                    if i == 8 and j == 3:
                        print("p")
                    ind = 0
                    if not quantized_frame.is_key_frame:
                        bit_stream += dictionary[str(abs(motion_vector[i][j][0]))]
                        bit_stream += dictionary[str(abs(motion_vector[i][j][1]))]
                    while ind < len(channel[i][j]) - 1:
                        item = channel[i][j][ind]
                        try:
                            bit_stream += dictionary[str(abs(item))]

                            if ind % 3 == 1:
                                bit_stream += "0" if item < 0 else "1"

                                bit_stream += (
                                    "0" if channel[i][j][ind + 1] == 0 else "1"
                                )
                                ind += 1

                        except Exception:
                            print("Word: " + str(abs(item)) + " not exist")

                        ind += 1

            bit_stream_for_all_channels += bit_stream
        return bit_stream_for_all_channels

    @staticmethod
    def _calculate_distance(array_1: np.ndarray, array_2: np.ndarray) -> float:
        return np.linalg.norm(np.array(array_1) - np.array(array_2))

    @staticmethod
    def check_blocks(channel):
        import matplotlib.ticker as plticker

        fig = plt.figure()

        ax3 = fig.add_subplot()
        myInterval = 8.0
        loc = plticker.MultipleLocator(base=myInterval)
        ax3.xaxis.set_major_locator(loc)
        ax3.yaxis.set_major_locator(loc)

        ax3.grid(which="major", axis="both", linestyle="-")
        ax3.imshow(channel, cmap=plt.get_cmap(name="gray"))
        plt.show()

    def motion_estimation(
        self, current_frame: Frame, reconstructed_frame: Frame,
    ) -> Frame:

        mv_for_channels = []
        channels = []

        reconstructed_frame_channels = reconstructed_frame.channels.list_channels

        for channel, reconstructed_channel in zip(
            current_frame.channels.list_channels, reconstructed_frame_channels
        ):

            width, height = channel.shape
            width_num = width // BLOCK_SIZE
            height_num = height // BLOCK_SIZE

            motion_vectors = np.zeros((width_num, height_num, 2)).astype(np.uint8)

            end_num = SEARCH_AREA // BLOCK_SIZE
            interval = (SEARCH_AREA - BLOCK_SIZE) // 2

            mask_image_1 = np.zeros((width + interval * 2, height + interval * 2))
            mask_image_1[
                : mask_image_1.shape[0] - interval * 2,
                : mask_image_1.shape[1] - interval * 2,
            ] = np.array(reconstructed_channel)

            predict_image = np.zeros(reconstructed_channel.shape)

            for i in range(width_num - 1):
                for j in range(height_num - 1):

                    temp_image = channel[
                        i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                        j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
                    ]
                    mask_image = mask_image_1[
                        i * BLOCK_SIZE : i * BLOCK_SIZE + SEARCH_AREA,
                        j * BLOCK_SIZE : j * BLOCK_SIZE + SEARCH_AREA,
                    ]

                    temp_res = self._calculate_distance(
                        mask_image[:BLOCK_SIZE, :BLOCK_SIZE], temp_image
                    )

                    for k in range(end_num):
                        for h in range(end_num):
                            temp_mask = mask_image[
                                k * BLOCK_SIZE : (k + 1) * BLOCK_SIZE,
                                h * BLOCK_SIZE : (h + 1) * BLOCK_SIZE,
                            ]

                            res = self._calculate_distance(temp_mask, temp_image)
                            if res < temp_res or k == 0 and h == 0:

                                temp_res = res
                                motion_vectors[i][j][0], motion_vectors[i][j][1] = k, h
                                predict_image[
                                    i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                                    j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
                                ] = temp_mask

            residual_frame = self.residual_compression(channel, predict_image)
            channels.append(np.array(residual_frame))
            mv_for_channels.append(np.array(motion_vectors))

        current_frame.channels.is_encoded = False
        current_frame.channels.luminosity = channels[0]
        current_frame.channels.chromaticCb = channels[1]
        current_frame.channels.chromaticCr = channels[2]
        current_frame.channels.mv_luminosity = mv_for_channels[0]
        current_frame.channels.mv_chromaticCb = mv_for_channels[1]
        current_frame.channels.mv_chromaticCr = mv_for_channels[2]

        return current_frame

    @staticmethod
    def residual_compression(original_frame, predicted_frame):
        return np.subtract(original_frame, predicted_frame)
