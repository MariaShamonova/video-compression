import dataclasses
import math

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
from repository import get_probabilities, HuffmanTree, reshape_frame, round_num


from sklearn.metrics import mean_squared_error
from scipy.fftpack import dct, idct


@dataclasses.dataclass
class Encoder:
    def encode_traditional_method(self, frame: Frame) -> Frame:

        all_dct_elements = []
        quantized_channels = []

        for idx, channel in enumerate(frame.channels.list_channels):

            X = reshape_frame(channel, BLOCK_SIZE)
            row, column = X.shape[0], X.shape[1]

            Y = [[[] for c in range(column)] for r in range(row)]

            for i in range(0,  row):
                for j in range(0, column):
                    dct_coeff = self.dct(X[i][j], BLOCK_SIZE)
                    # if i == 0 and j == 0:
                    #     self.dct_output(dct_coeff)

                    quantinization_coeff = self.quantization(
                        dct_coeff,
                        MATRIX_QUANTIZATION_CHROMATIC
                        if idx > 0
                        else MATRIX_QUANTIZATION,
                    )
                    Y[i][j] = quantinization_coeff

                    # sequence_coeff = self.zig_zag_transform(quantinization_coeff)
                    #
                    # series_value_coeff = self.separate_pair(sequence_coeff)
                    # Y[i][j] = series_value_coeff
                    # all_dct_elements.append(abs(Y[i][j][1]))
                    # all_dct_elements.extend([abs(Y[i][j][index])
                    #                          for index in range(1, len(Y[i][j]) - 1)])

            # dict_Haffman = self.entropy_encoder(all_dct_elements)
            # bit_stream = self.transform_to_bit_stream(dict_Haffman, Y)

            # return bit_stream, dict_Haffman, frame
            quantized_channels.append(np.array(Y))

        frame.channels.is_encoded = True
        frame.channels.luminosity = quantized_channels[0]
        frame.channels.chromaticCr = quantized_channels[1]
        frame.channels.chromaticCb = quantized_channels[2]

        return frame

    def encodeNN(self, frame: Frame) -> Frame:
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

        frame.channels.luminosity = layers_pipeline(
            frame.channels.luminosity.unsqueeze(0)
        )
        frame.channels.chromaticCb = layers_pipeline(
            frame.channels.chromaticCb.unsqueeze(0)
        )
        frame.channels.chromaticCr = layers_pipeline(
            frame.channels.chromaticCr.unsqueeze(0)
        )
        frame.channels.is_encoded = True

        return frame

    def encode_I_frame(self, frame: Frame, method) -> Frame:

        if method == 0:
            return self.encode_traditional_method(frame=frame)
        else:
            return self.encodeNN(frame=frame)

    def encode_B_frame(self, frame: Frame, reconstructed_frame: Frame, method) -> Frame:
        assert frame.is_key_frame is False

        residual_frame = self.motion_estimation(frame, reconstructed_frame)
        if method == 0:
            encoded_frame = self.encode_traditional_method(frame=residual_frame)
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

    @staticmethod
    def _get_matrix_A(BLOCK_SIZE_FOR_DCT):
        A = [
            [
                np.round(
                    math.sqrt((1 if (i == 0) else 2) / BLOCK_SIZE_FOR_DCT)
                    * math.cos(((2 * j + 1) * i * math.pi) / (2 * BLOCK_SIZE_FOR_DCT)),
                    3,
                )
                for j in range(0, BLOCK_SIZE_FOR_DCT)
            ]
            for i in range(0, BLOCK_SIZE_FOR_DCT)
        ]
        return A

    def dct(self, X, BLOCK_SIZE_FOR_DCT):
        A = self._get_matrix_A(BLOCK_SIZE_FOR_DCT)

        return np.array(A).dot(X).dot(np.array(A).transpose())

    def dct_output(self, block):
        plt.figure()
        plt.imshow(block, cmap='gray', interpolation='nearest')
        plt.colorbar(shrink=0.5)
        plt.title("DCT transform of selected Region")
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
        seq = seq[: i + 1]

        for i in range(0, len(seq)):

            if seq[i] != 0:
                is_latest_element = "ЕОВ" if i == (len(seq) - 1) else 0
                transform_seq.extend([63, seq[i], is_latest_element])

                count_zero = 0
            else:
                count_zero += 1

        return transform_seq

    @staticmethod
    def entropy_encoder(blocks):

        probability = get_probabilities(blocks)
        # codewars = algorithmHaffman(probability)

        tree = HuffmanTree(probability)
        codewars = tree.get_code()

        # print(codewars)
        # transformToBitStream(codewars, blocks)

        return codewars

    @staticmethod
    def transform_to_bit_stream(codewars, Y):
        bit_stream = ""

        r = len(Y)
        c = len(Y[0])

        for i in range(r):
            for j in range(c):
                ind = 0

                while ind < len(Y[i][j]) - 1:
                    item = Y[i][j][ind]
                    try:
                        bit_stream += codewars[str(abs(item))]

                        if ind % 3 == 1:
                            bit_stream += "0" if item < 0 else "1"

                            bit_stream += "0" if Y[i][j][ind + 1] == 0 else "1"
                            ind += 1

                    except Exception:
                        print("Word: " + str(abs(item)) + " not exist")

                    ind += 1

        return bit_stream

    @staticmethod
    def _calculate_distance(array_1: np.ndarray, array_2: np.ndarray) -> float:
        return np.linalg.norm(np.array(array_1) - np.array(array_2))
        # return mean_squared_error(array_1, array_2)

    def motion_estimation(
        self, current_frame: Frame, reconstructed_frame: Frame,
    ) -> Frame:

        mv_for_channels = []
        channels = []
        reconstructed_frame_channels = reconstructed_frame.channels.list_channels

        for channel, reconstructed_channels in zip(
            current_frame.channels.list_channels, reconstructed_frame_channels
        ):
            width, height = channel.shape
            width_num = width // BLOCK_SIZE
            height_num = height // BLOCK_SIZE

            motion_vectors = [
                [[0, 0] for _ in range(height_num)] for _ in range(width_num)
            ]

            end_num = SEARCH_AREA // BLOCK_SIZE
            interval = (SEARCH_AREA - BLOCK_SIZE) // 2

            mask_image_1 = np.zeros(
                (width + interval * 2, height + interval * 2)
            ).astype("uint8")
            mask_image_1[
                : mask_image_1.shape[0] - interval * 2,
                : mask_image_1.shape[1] - interval * 2,
            ] = np.array(reconstructed_channels)

            predict_image = np.zeros(reconstructed_channels.shape).astype("uint8")

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
                            if res <= temp_res:
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
