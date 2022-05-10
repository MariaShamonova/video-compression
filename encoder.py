import dataclasses
import math

import numpy as np
from constants import MATRIX_QUANTIZATION
from repository import get_probabilities, HuffmanTree
from sklearn.metrics import mean_squared_error

@dataclasses.dataclass
class Encoder:

    @staticmethod
    def transform_rgb_to_y(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    @staticmethod
    def transform_rgb_to_cbcr(frame_y, frame):
        r, c = int(np.array(frame).shape[0] / 4), int(np.array(frame).shape[1] / 4)
        blocks = np.zeros((r, c, 2))

        blocks[0, 1] = [1, 2]
        for i in range(0, r):
            for j in range(0, c):
                x = 2 * i + 1
                y = 2 * j + 1
                blocks[i][j] = [0.713 * (frame[x][y][0] - frame_y[x])
                [y], 0.564 * (frame[x][y][2] - frame_y[x][y])]
        return blocks

    @staticmethod
    def _get_matrix_A(N):
        A = [[np.round(math.sqrt((1 if (i == 0) else 2) / N) * math.cos(((2 * j + 1) * i * math.pi) / (2 * N)), 3)
              for j in range(0, N)] for i in range(0, N)]
        return A

    def dct(self, X, N):
        A = self._get_matrix_A(N)

        return np.array(A).dot(X).dot(np.array(A).transpose())

    def quantization(self, Y):
        quantization_coeff = np.round(
            (np.divide(np.array(Y), MATRIX_QUANTIZATION))).astype(int)
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

            diag = [block[len(block) - index + i][len(block) - i - 1]
                    for i in range(len(slice))]

            if len(diag) % 2:
                diag.reverse()
            zigzag += diag

        return zigzag

    @staticmethod
    def entropy_encoder(self):
        print("EntropyEncoder")

    @staticmethod
    def separate_pair(seq):
        transformSeq = []
        while (seq[len(seq) - 1] == 0):
            seq.pop(len(seq) - 1)

        count_zero = 0

        for i in range(0, len(seq)):

            if (seq[i] != 0):
                isLatestElement = 'ЕОВ' if i == (len(seq) - 1) else 0
                transformSeq.extend([count_zero, seq[i], isLatestElement])

                count_zero = 0
            else:
                count_zero += 1

        return transformSeq

    @staticmethod
    def entropy_encoding(blocks):

        probability = get_probabilities(blocks)
        # codewars = algorithmHaffman(probability)

        tree = HuffmanTree(probability)
        codewars = tree.get_code()

        # print(codewars)
        # transformToBitStream(codewars, blocks)

        return codewars

    @staticmethod
    def transform_to_bit_stream(codewars, Y):
        bit_stream = ''

        r = len(Y)
        c = len(Y[0])

        for i in range(r):
            for j in range(c):
                ind = 0

                while ind < len(Y[i][j]) - 1:
                    item = Y[i][j][ind]
                    try:
                        bit_stream += codewars[str(abs(item))]

                        if (ind % 3 == 1):
                            bit_stream += ('0' if item < 0 else '1')

                            bit_stream += ('0' if Y[i][j][ind + 1] == 0 else '1')
                            ind += 1

                    except Exception:
                        print("Word: " + str(abs(item)) + ' not exist')

                    ind += 1

        return bit_stream

    @staticmethod
    def _calculate_distance(array_1: np.ndarray, array_2: np.ndarray) -> float:
        return np.linalg.norm(np.array(array_1) - np.array(array_2))
        # return mean_squared_error(array_1, array_2)

    def motion_estimation(self, capture_1_Y, capture_2_Y, width, height, block_sizes, search_areas):

        width_num = width // block_sizes
        height_num = height // block_sizes
        # Number of motion vectors
        vet_nums = width_num * height_num
        # Used to save motion vectors and coordinate values
        motion_vectors = []
        # Give null value first
        motion_vectors = [[0, 0] for _ in range(vet_nums)]

        motion_vectors_for_draw = [[[0, 0, 0, 0] for j in range(width_num)] for i in range( height_num)]

        similarity = 0
        num = 0
        end_num = search_areas // block_sizes

        # Calculation interval, used to make up 0
        interval = (search_areas - block_sizes) // 2
        # Construct a template image and add 0 to the previous frame image
        mask_image_1 = np.zeros((height + interval * 2,  width + interval * 2))

        mask_image_1[:mask_image_1.shape[0] - interval*2 , :mask_image_1.shape[1] - interval*2] = np.array(capture_1_Y)

        mask_width, mask_height = mask_image_1.shape

        predict_image = np.zeros(capture_1_Y.shape)
        print([[[0, 0] for j in range(height_num)] for i in range(width_num)])
        #     count = 0
        for i in range(height_num):
            for j in range(width_num):
                #             count += 1
                #         print(f'==================i:{i}=j:{j}==count:{count}=====================')
                temp_image = capture_2_Y[i * block_sizes:(i + 1) * block_sizes, j * block_sizes:(j + 1) * block_sizes]
                mask_image = mask_image_1[i * block_sizes:i * block_sizes + search_areas,
                             j * block_sizes:j * block_sizes + search_areas]
                #  Given initial value for comparison
                temp_res = self._calculate_distance(mask_image[:block_sizes, :block_sizes], temp_image)
                for k in range(end_num):
                    for h in range(end_num):
                        temp_mask = mask_image[k * block_sizes:(k + 1) * block_sizes,
                                    h * block_sizes:(h + 1) * block_sizes]

                        res = self._calculate_distance(temp_mask, temp_image)
                        if res <= temp_res:
                            temp_res = res
                            motion_vectors[i * j][0], motion_vectors[i * j ][1] = k, h

                            motion_vectors_for_draw[i][j][0], motion_vectors_for_draw[i][j][1], motion_vectors_for_draw[i][j][2], motion_vectors_for_draw[i][j][3] = \
                                i * search_areas + search_areas//2, j*search_areas + search_areas//2, \
                                     i*search_areas + k*block_sizes + search_areas//2,  j*search_areas + h*block_sizes + search_areas//2
                            predict_image[i * block_sizes:(i + 1) * block_sizes,
                            j * block_sizes:(j + 1) * block_sizes] = temp_mask
        #                         print(motion_vectors_for_draw[i*j+j])
        return np.array(predict_image), np.array(motion_vectors), np.array(motion_vectors_for_draw)





