import dataclasses
import math
import time

from matplotlib import pyplot as plt
from scipy.fftpack import idct
import cv2
import numpy as np

from constants import (
    MATRIX_QUANTIZATION,
    MATRIX_QUANTIZATION_CHROMATIC,
    BLOCK_SIZE,
)
from frame import Channels, Frame
from repository import reshape_frame, concat_blocks, append_num


@dataclasses.dataclass
class Decoder:
    def _decode_traditional(self, bit_stream: str, dictionary: dict, is_key_frame: bool, method: int, width: int, height: int) -> Frame:
        dequantized_channels = []
        channels, motion_vectors = self.entropy_decoder(bit_stream, dictionary, is_key_frame, method, width, height)

        for idx, channel in enumerate(channels):

            frame_shape = np.array(channels[idx]).shape
            rows, columns = frame_shape[0], frame_shape[1]

            Y = [[[] for c in range(columns)] for r in range(int(rows))]

            for i in range(rows):
                for j in range(columns):

                    Y[i][j] = self.inverse_zig_zag_transform(channels[idx][i][j])

                    Y[i][j] = self.dequantization(
                        Y[i][j],
                        MATRIX_QUANTIZATION_CHROMATIC
                        if idx > 0
                        else MATRIX_QUANTIZATION,
                    )

                    Y[i][j] = self.idct(Y[i][j]).astype(np.uint8)

            dequantized_channels.append(concat_blocks(np.array(Y)))

        if is_key_frame:

            channels = Channels(
                is_encoded=False,
                luminosity=dequantized_channels[0],
                chromaticCr=dequantized_channels[1],
                chromaticCb=dequantized_channels[2]
            )
        else:
            channels = Channels(
                is_encoded=False,
                luminosity=dequantized_channels[0],
                chromaticCr=dequantized_channels[1],
                chromaticCb=dequantized_channels[2],
                mv_luminosity=motion_vectors[0],
                mv_chromaticCr=motion_vectors[1],
                mv_chromaticCb=motion_vectors[2]
            )

        encoded_frame = Frame(channels=channels, is_key_frame=is_key_frame, width=width, height=height)
        return encoded_frame

    @staticmethod
    def _decode_nn(encoded_frame: Frame) -> Frame:
        import torch
        import torch.nn as nn

        ker = (
            torch.tensor([[0, 0, 0], [0, 0, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]])
            .type(torch.FloatTensor)
            .repeat(1, 1, 1)
        )

        decon = nn.ConvTranspose2d(1, 1, 3, stride=1, bias=False)
        decon.weight = torch.nn.Parameter(ker.unsqueeze(0))

        encoded_frame.channels.luminosity = (
            decon(encoded_frame.channels.luminosity)
            .mul_(255.0)
            .clamp_(0.0, 255.0)
            .squeeze(0)
            .squeeze(0)
            .byte()
            .cpu()
            .numpy()
        )
        encoded_frame.channels.chromaticCr = (
            decon(encoded_frame.channels.chromaticCr)
            .mul_(255.0)
            .clamp_(0.0, 255.0)
            .squeeze(0)
            .squeeze(0)
            .byte()
            .cpu()
            .numpy()
        )
        encoded_frame.channels.chromaticCb = (
            decon(encoded_frame.channels.chromaticCb)
            .mul_(255.0)
            .clamp_(0.0, 255.0)
            .squeeze(0)
            .squeeze(0)
            .byte()
            .cpu()
            .numpy()
        )
        encoded_frame.channels.is_encoded = False
        upsampled_frame = encoded_frame.upsample_image()
        decoded_frame_channels = Frame(
            frame=upsampled_frame,
            is_key_frame=encoded_frame.is_key_frame,
            width=encoded_frame.width,
            height=encoded_frame.height,
            method=1
        ).channels

        encoded_frame.channels.luminosity = decoded_frame_channels.luminosity
        encoded_frame.channels.chromaticCb = decoded_frame_channels.chromaticCb
        encoded_frame.channels.chromaticCr = decoded_frame_channels.chromaticCr

        return encoded_frame

    def decode(self, bit_stream: str, dictionary: dict, is_key_frame: bool, method: int, width: int, height: int) -> Frame:
        # assert encoded_frame.channels.is_encoded is True

        if method == 0:
            return self._decode_traditional(bit_stream, dictionary, is_key_frame, method, width, height)
        else:
            return self._decode_nn(bit_stream)

    def decode_B_frame(self, decoded_frame: Frame, reconstructed_frame: Frame, method):

        reconstructed_channels = []
        num_chanels = len(decoded_frame.channels.list_channels)
        for chanel_index in range(num_chanels):


            decoded_frame_shape = decoded_frame.channels.list_channels[
                chanel_index
            ].shape
            width, height = decoded_frame_shape[0], decoded_frame_shape[1]

            micro_blocks = reshape_frame(
                reconstructed_frame.channels.list_channels[chanel_index], BLOCK_SIZE
            )
            predicted_image = reshape_frame(
                np.zeros(
                    reconstructed_frame.channels.list_channels[chanel_index].shape
                ),
                BLOCK_SIZE,
            )

            width_num = int(width // BLOCK_SIZE)
            height_num = int(height // BLOCK_SIZE)

            for i in range(width_num):
                for j in range(height_num):

                    current_vector = decoded_frame.channels.list_motion_vectors[
                        chanel_index
                    ][i][j]
                    try:
                        predicted_image[i][j] = micro_blocks[i + current_vector[0]][
                            j + current_vector[1]
                        ]
                    except:
                        print(i, j)

            predicted_image = concat_blocks(predicted_image)

            reconstructed_channels.append(
                self.residual_decompression(
                    predicted_image, decoded_frame.channels.list_channels[chanel_index]
                )
            )

        decoded_frame.channels.is_encoded = False
        decoded_frame.channels.luminosity = reconstructed_channels[0]
        decoded_frame.channels.chromaticCr = reconstructed_channels[1]
        decoded_frame.channels.chromaticCb = reconstructed_channels[2]

        return decoded_frame

    def idct(self, Y):
        return idct(idct(Y, axis=0, norm='ortho'), axis=1, norm='ortho')

    def dequantization(self, Y, coefficient):
        return np.multiply(np.array(Y), coefficient)

    def inverse_zig_zag_transform(self, zigzag):
        blocks_8x8 = np.zeros((BLOCK_SIZE, BLOCK_SIZE))

        for index in range(1, 9):
            slice = [i[:index] for i in blocks_8x8[:index]]
            i = 0 if index % 2 == 0 else index - 1
            j = 0 if index % 2 == 1 else index - 1
            for ind in range(len(slice)):

                blocks_8x8[i][j] = zigzag[0]
                zigzag.pop(0)

                i = i + (1 if index % 2 == 0 else -1)
                j = j + (1 if index % 2 == 1 else -1)

        for index in reversed(range(1, 8)):

            slice = [i[:index] for i in blocks_8x8[:index]]
            i = (
                BLOCK_SIZE - len(slice)
                if index % 2 == 0
                else BLOCK_SIZE - 1
            )
            j = (
                BLOCK_SIZE - len(slice)
                if index % 2 == 1
                else BLOCK_SIZE - 1
            )

            for ind in range(len(slice)):
                blocks_8x8[i][j] = zigzag[0]
                zigzag.pop(0)
                i = i + (1 if index % 2 == 0 else -1)
                j = j + (1 if index % 2 == 1 else -1)

        return blocks_8x8



    def entropy_decoder(self, bit_stream, dictionary, is_key_frame, method, width, height):
        values = []
        bites = ""
        num_values = 0

        dictionary = dict((v, k) for k, v in dictionary.items())

        rows = height // 8
        columns = width // 8
        blocks = [[[] for i in range(columns)] for j in range(rows)]
        motion_vectors = [[[0, 0] for i in range(columns)] for j in range(rows)]
        temp_channels = []
        temp_motion_vectors = []
        num_channel = 0
        num_channel_temp = 0
        r = 0
        c = 0

        i = 0
        while i <= len(bit_stream):

            try:
                bites += bit_stream[i]
                value = dictionary[bites]

                if num_values == 0 and not is_key_frame:
                    motion_vectors[r][c][0] = int(value)
                    num_values += 1
                elif num_values == 1 and not is_key_frame:
                    motion_vectors[r][c][1] = int(value)
                    num_values += 1
                elif num_values == 2 and not is_key_frame or num_values == 0 and is_key_frame:
                    values.extend([0 for z in range(int(value))])
                    num_values += 1
                elif num_values == 3 and not is_key_frame or num_values == 1 and is_key_frame:
                    values.append(
                        int(("" if int(bit_stream[i+1]) else "-") + value)
                    )
                    num_values = 0
                    if not is_key_frame and bit_stream[i+2] != "1":
                        num_values = 2

                    if bit_stream[i+2] == "1":
                        values.extend(
                            [
                                0
                                for z in range(
                                    int(
                                        BLOCK_SIZE * BLOCK_SIZE
                                        - len(values)
                                    )
                                )
                            ]
                        )

                        blocks[r][c] = values



                        if c == columns - 1 and r < rows:

                            r += 1


                        if r == rows and c == columns - 1:
                            temp_channels.append(blocks)
                            temp_motion_vectors.append(motion_vectors)
                            num_channel += 1
                            bit_stream = bit_stream[i+3:]
                            i = -3

                            if num_channel == 1 and method == 0:
                                rows = append_num(height // 2) // 8
                                columns = append_num(width // 2) // 8
                                print(bit_stream[:100])

                            values = []
                            blocks = [[[] for i in range(columns)] for j in range(rows)]
                            motion_vectors = [[[0, 0] for i in range(columns)] for j in range(rows)]
                            r = 0

                        c = c + 1 if c < columns - 1 else 0

                        values = []



                    i += 2

                bites = ""
            except Exception:
                pass

            i += 1

        if is_key_frame:
            return temp_channels, []
        else:
            return temp_channels, temp_motion_vectors


    @staticmethod
    def restruct_image(
        width,
        height,
        block_sizes,
        search_areas,
        motion_vectors,
        residual_image,
        pre_frame,
    ):

        width_num = width // block_sizes
        height_num = height // block_sizes
        vet_nums = width_num * height_num

        end_num = search_areas // block_sizes

        # Calculation interval, used to make up 0
        interval = (search_areas - block_sizes) // 2
        # Construct a template image and add 0 to the previous frame image
        mask_image_1 = np.zeros((width + interval * 2, height + interval * 2))
        mask_image_1[
            : mask_image_1.shape[0] - interval * 2,
            : mask_image_1.shape[1] - interval * 2,
        ] = pre_frame

        restruct_image = np.zeros((width, height))

        for i in range(height_num):
            for j in range(width_num):
                temp_image = residual_image[
                    i * block_sizes : (i + 1) * block_sizes,
                    j * block_sizes : (j + 1) * block_sizes,
                ]
                mask_image = mask_image_1[
                    i * block_sizes : i * block_sizes + search_areas,
                    j * block_sizes : j * block_sizes + search_areas,
                ]
                #  Given initial value for comparison

                k, h = motion_vectors[(i * j) + j][0], motion_vectors[(i * j) + j][1]
                restruct_image[
                    i * block_sizes : (i + 1) * block_sizes,
                    j * block_sizes : (j + 1) * block_sizes,
                ] = (
                    temp_image
                    + mask_image[
                        k * block_sizes : (k + 1) * block_sizes,
                        h * block_sizes : (h + 1) * block_sizes,
                    ]
                )
        return np.array(restruct_image, dtype=np.uint8)

    @staticmethod
    def residual_decompression(original_frame, predicted_frame):
        return np.add(original_frame, predicted_frame)
