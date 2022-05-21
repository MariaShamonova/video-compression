import dataclasses
import math
import time

import numpy as np

from constants import (
    MATRIX_QUANTIZATION,
    MATRIX_QUANTIZATION_CHROMATIC,
    BLOCK_SIZE,
)
from frame import Channels, Frame
from repository import reshape_frame
from repository import concat_blocks


@dataclasses.dataclass
class Decoder:
    def _decode_traditional(self, encoded_frame: Frame) -> Frame:
        dequantized_channels = []
        # blocks = self.entropy_decoder(bit_stream, codewars, BLOCK_SIZE_FOR_DCT, shape)

        for idx, channel in enumerate(encoded_frame.channels.list_channels):

            frame_shape = encoded_frame.channels.list_channels[idx].shape
            rows, columns = frame_shape[0], frame_shape[1]

            Y = [[[] for c in range(columns)] for r in range(int(rows))]

            for i in range(rows):
                for j in range(columns):
                    # Y[i][j] = self.inverse_zig_zag_transform(blocks[i][j], 8)
                    Y[i][j] = encoded_frame.channels.list_channels[idx][i][j]
                    Y[i][j] = self.dequantization(
                        Y[i][j],
                        MATRIX_QUANTIZATION,
                    )
                    Y[i][j] = self.idct(Y[i][j])

            dequantized_channels.append(concat_blocks(Y))

        encoded_frame.channels.luminosity = dequantized_channels[0]
        encoded_frame.show_luminosity_channel()
        encoded_frame.channels.chromaticCr = dequantized_channels[1]
        encoded_frame.channels.chromaticCb = dequantized_channels[2]

        encoded_frame.channels.is_encoded = False

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

        decoded_frame_channels = Frame(
            frame=encoded_frame.upsample_image(),
            is_key_frame=encoded_frame.is_key_frame,
            width=encoded_frame.width,
            height=encoded_frame.height,
        ).channels

        encoded_frame.channels.luminosity = decoded_frame_channels.luminosity
        encoded_frame.channels.chromaticCb = decoded_frame_channels.chromaticCb
        encoded_frame.channels.chromaticCr = decoded_frame_channels.chromaticCr

        return encoded_frame

    def decode(self, encoded_frame: Frame, method) -> Frame:
        assert encoded_frame.channels.is_encoded is True

        if method == 0:
            return self._decode_traditional(encoded_frame)
        else:
            return self._decode_nn(encoded_frame)

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
        decoded_frame.channels.chromaticCb = reconstructed_channels[1]
        decoded_frame.channels.chromaticCr = reconstructed_channels[2]

        return decoded_frame

        # predict_image, motion_vectors, motion_vectors_for_draw = self.motion_estimation(reconstructed_frame, frame_y,
        #                                                                                 width, height, BLOCK_SIZE,
        #                                                                                 SEARCH_AREA)
        # reconstructed_frame = self.residual_decompression(frame, residual_frame)

        # return self.encode(frame=residual_frame)

    @staticmethod
    def _get_matrix_A():
        A = [
            [
                np.round(
                    math.sqrt((1 if (i == 0) else 2) / BLOCK_SIZE)
                    * math.cos(((2 * j + 1) * i * math.pi) / (2 * BLOCK_SIZE)),
                    3,
                )
                for j in range(0, BLOCK_SIZE)
            ]
            for i in range(0, BLOCK_SIZE)
        ]
        return A

    def idct(self, Y):
        A = self._get_matrix_A()

        return np.array(A).transpose().dot(Y).dot(np.array(A))

    def dequantization(self, Y, coefficient):
        return np.multiply(np.array(Y), coefficient)

    def inverse_zig_zag_transform(self, zigzag, BLOCK_SIZE_FOR_DCT):
        blocks_8x8 = np.zeros((BLOCK_SIZE_FOR_DCT, BLOCK_SIZE_FOR_DCT))

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
                BLOCK_SIZE_FOR_DCT - len(slice)
                if index % 2 == 0
                else BLOCK_SIZE_FOR_DCT - 1
            )
            j = (
                BLOCK_SIZE_FOR_DCT - len(slice)
                if index % 2 == 1
                else BLOCK_SIZE_FOR_DCT - 1
            )

            for ind in range(len(slice)):
                blocks_8x8[i][j] = int(zigzag[0])
                zigzag.pop(0)
                i = i + (1 if index % 2 == 0 else -1)
                j = j + (1 if index % 2 == 1 else -1)

        return blocks_8x8.astype(int)

    @staticmethod
    def entropy_decoder(bit_stream, codewars, BLOCK_SIZE_FOR_DCT, shape):
        values = []
        bites = ""

        codewars = dict((v, k) for k, v in codewars.items())

        countValues = 0
        i = 0

        rows = int(shape[0] / 8)
        columns = int(shape[1] / 8)

        blocks = [[[] for i in range(columns)] for j in range(rows)]

        r = 0
        c = 0

        start_time = time.time()
        while i <= len(bit_stream):
            try:
                bites += bit_stream[i]
                value = codewars[bites]

                if countValues == 0:

                    values.extend([0 for z in range(int(value))])

                    countValues = 1
                elif countValues == 1:
                    values.append(
                        int(("" if int(bit_stream[i + 1]) else "-") + value)
                    )  # Значение

                    if bit_stream[i + 2] == "1":
                        values.extend(
                            [
                                0
                                for z in range(
                                    int(
                                        BLOCK_SIZE_FOR_DCT * BLOCK_SIZE_FOR_DCT
                                        - len(values)
                                    )
                                )
                            ]
                        )

                        blocks[r][c] = values

                        if c == columns - 1 and r < rows - 1:
                            r += 1

                        c = c + 1 if c < columns - 1 else 0

                        values = []
                    countValues = 0
                    i += 2
                bites = ""

            except Exception:
                pass

            i += 1
        print("--- %s seconds ---" % (time.time() - start_time))
        return blocks

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

    def residual_decompression(self, original_frame, predicted_frame):

        return np.add(original_frame, predicted_frame)
