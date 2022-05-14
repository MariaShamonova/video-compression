import dataclasses
import math
import time

import numpy as np
from constants import MATRIX_QUANTIZATION, BLOCK_SIZE, SEARCH_AREA, BLOCK_SIZE_FOR_DCT


@dataclasses.dataclass
class Decoder:

    def decode(self, bit_stream, codewars, shape):

        blocks = self.entropy_decoder(bit_stream, codewars, BLOCK_SIZE_FOR_DCT, shape)

        r = len(blocks)
        c = len(blocks[0])

        Y = [[[] for c in range(int(shape[1] / BLOCK_SIZE_FOR_DCT))]
             for r in range(int(shape[0] / BLOCK_SIZE_FOR_DCT))]

        for i in range(r):
            for j in range(c):

                Y[i][j] = self.inverse_zig_zag_transform(blocks[i][j], 8)

                Y[i][j] = self.dequantization(Y[i][j])

                Y[i][j] = self.idct(Y[i][j], BLOCK_SIZE_FOR_DCT)

        Y = self.concat_blocks(Y)
        return Y

    def decode_B_frame(self, frame,  residual_frame):
        height, width, index = frame.shape
        frame_y = self.transform_rgb_to_y(frame)
        predict_image, motion_vectors, motion_vectors_for_draw = self.motion_estimation(reconstructed_frame, frame_y,
                                                                                        width, height, BLOCK_SIZE,
                                                                                        SEARCH_AREA)
        reconstructed_frame = self.residual_decompression(frame, residual_frame)

        return self.encode(frame=residual_frame)

    @staticmethod
    def _get_matrix_A():
        A = [[np.round(math.sqrt((1 if (i == 0) else 2) / BLOCK_SIZE_FOR_DCT) * math.cos(((2 * j + 1) * i * math.pi) / (2 * BLOCK_SIZE_FOR_DCT)), 3)
              for j in range(0, BLOCK_SIZE_FOR_DCT)] for i in range(0, BLOCK_SIZE_FOR_DCT)]
        return A


    def idct(self, Y):
        A = self._get_matrix_A()

        return np.array(A).transpose().dot(Y).dot(np.array(A))

    def dequantization(self, Y):
        return np.multiply(np.array(Y), MATRIX_QUANTIZATION)

    def inverse_zig_zag_transform(self, zigzag, BLOCK_SIZE_FOR_DCT):
        blocks_8x8 = np.zeros((BLOCK_SIZE_FOR_DCT, BLOCK_SIZE_FOR_DCT))

        for index in range(1, 9):
            slice = [i[:index] for i in blocks_8x8[:index]]
            i = (0 if index % 2 == 0 else index - 1)
            j = (0 if index % 2 == 1 else index - 1)
            for ind in range(len(slice)):

                blocks_8x8[i][j] = zigzag[0]

                zigzag.pop(0)
                i = i + (1 if index % 2 == 0 else -1)
                j = j + (1 if index % 2 == 1 else -1)

        for index in reversed(range(1, 8)):

            slice = [i[:index] for i in blocks_8x8[:index]]
            i = (BLOCK_SIZE_FOR_DCT - len(slice) if index % 2 == 0 else BLOCK_SIZE_FOR_DCT - 1)
            j = (BLOCK_SIZE_FOR_DCT - len(slice) if index % 2 == 1 else BLOCK_SIZE_FOR_DCT - 1)

            for ind in range(len(slice)):
                blocks_8x8[i][j] = int(zigzag[0])
                zigzag.pop(0)
                i = i + (1 if index % 2 == 0 else -1)
                j = j + (1 if index % 2 == 1 else -1)

        return blocks_8x8.astype(int)

    @staticmethod
    def entropy_decoder(bit_stream, codewars, BLOCK_SIZE_FOR_DCT, shape):
        values = []
        bites = ''

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

                if (countValues == 0):

                    values.extend([0 for z in range(int(value))])

                    countValues = 1
                elif (countValues == 1):
                    values.append(
                        int(('' if int(bit_stream[i + 1]) else '-') + value))  # Значение

                    if (bit_stream[i + 2] == '1'):
                        values.extend([0 for z in range(int(BLOCK_SIZE_FOR_DCT * BLOCK_SIZE_FOR_DCT - len(values)))])

                        blocks[r][c] = values

                        if c == columns - 1 and r < rows - 1:
                            r += 1

                        c = (c + 1 if c < columns - 1 else 0)


                        values = []
                    countValues = 0
                    i += 2
                bites = ''

            except Exception:
                pass

            i += 1
        print("--- %s seconds ---" % (time.time() - start_time))
        return blocks

    @staticmethod
    def concat_blocks(blocks):
        blocks = np.array(blocks)
        frame = []

        rows = blocks.shape[0]

        for i in range(rows):
            frame.append(np.concatenate((blocks[i]), axis=1))

        frame = np.concatenate(frame, axis=0)

        return np.array(frame)

    @staticmethod
    def restruct_image(width, height, block_sizes, search_areas, motion_vectors, residual_image, pre_frame):

        width_num = width // block_sizes
        height_num = height // block_sizes
        vet_nums = width_num * height_num

        end_num = search_areas // block_sizes

        # Calculation interval, used to make up 0
        interval = (search_areas - block_sizes) // 2
        # Construct a template image and add 0 to the previous frame image
        mask_image_1 = np.zeros((width + interval * 2, height + interval * 2))
        mask_image_1[:mask_image_1.shape[0] - interval*2, :mask_image_1.shape[1] - interval*2] = pre_frame

        restruct_image = np.zeros((width, height))

        for i in range(height_num):
            for j in range(width_num):
                temp_image = residual_image[i * block_sizes:(i + 1) * block_sizes,
                             j * block_sizes:(j + 1) * block_sizes]
                mask_image = mask_image_1[i * block_sizes:i * block_sizes + search_areas,
                             j * block_sizes:j * block_sizes + search_areas]
                #  Given initial value for comparison

                k, h = motion_vectors[(i * j) + j][0], motion_vectors[(i * j) + j][1]
                restruct_image[i * block_sizes:(i + 1) * block_sizes,
                j * block_sizes:(j + 1) * block_sizes] = temp_image + mask_image[k * block_sizes:(k + 1) * block_sizes,
                                                                      h * block_sizes:(h + 1) * block_sizes]
        return np.array(restruct_image, dtype=np.uint8)

    def residual_decompression(self, original_frame, predicted_frame):
        return np.add(original_frame, predicted_frame)


