import dataclasses
import math

import numpy as np
from constants import MATRIX_QUANTIZATION, BLOCK_SIZE, SEARCH_AREA, BLOCK_SIZE_FOR_DCT
from repository import get_probabilities, HuffmanTree
from reshapeFrame import reshapeFrame
from sklearn.metrics import mean_squared_error
from scipy.fftpack import dct, idct


@dataclasses.dataclass
class Encoder:

    def encode(self, frame):

        X = reshapeFrame(frame, BLOCK_SIZE_FOR_DCT)
        width, height = X.shape[0], X.shape[1]

        Y = [[[] for c in range(height)] for r in range(width)]
        all_dct_elements = []

        for i in range(0, width):
            for j in range(0, height):
                dct_coeff = self.dct(X[i][j], BLOCK_SIZE_FOR_DCT)
                quantinization_coeff = self.quantization(dct_coeff)
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
        return Y

    def encode_I_frame(self, frame):
        print(frame.shape)
        frame_y = self.transform_rgb_to_y(frame)

        return self.encode(frame=frame_y)

    def encode_B_frame(self, frame, reconstructed_frame):
        height, width, index = frame.shape

        frame_y = self.transform_rgb_to_y(frame)
        predict_image, motion_vectors, motion_vectors_for_draw = self.motion_estimation(reconstructed_frame, frame_y,
                                                                                           width, height, BLOCK_SIZE,
                                                                                           SEARCH_AREA)
        residual_frame = self.residual_compression(frame_y, predict_image)
        return self.encode(frame=residual_frame), motion_vectors, predict_image, residual_frame

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
    def _get_matrix_A(BLOCK_SIZE_FOR_DCT):
        A = [[np.round(math.sqrt((1 if (i == 0) else 2) / BLOCK_SIZE_FOR_DCT) * math.cos(((2 * j + 1) * i * math.pi) / (2 * BLOCK_SIZE_FOR_DCT)), 3)
              for j in range(0, BLOCK_SIZE_FOR_DCT)] for i in range(0, BLOCK_SIZE_FOR_DCT)]
        return A

    def dct(self, X, BLOCK_SIZE_FOR_DCT):
        A = self._get_matrix_A(BLOCK_SIZE_FOR_DCT)

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
        seq = seq[:i + 1]

        for i in range(0, len(seq)):

            if seq[i] != 0:
                is_latest_element = 'ЕОВ' if i == (len(seq) - 1) else 0
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

    def motion_estimation(self, reconstructed_image, original_frame, width, height, block_sizes, search_areas):

        width_num = width // block_sizes
        height_num = height // block_sizes
        # Number of motion vectors
        vet_nums = width_num * height_num
        # Used to save motion vectors and coordinate values
        motion_vectors = []
        # Give null value first
        motion_vectors = [[[0, 0] for _ in range(width_num)] for _ in range(height_num)]

        motion_vectors_for_draw = [[[0, 0, 0, 0] for j in range(width_num)] for i in range( height_num)]

        similarity = 0
        num = 0
        end_num = search_areas // block_sizes

        # Calculation interval, used to make up 0
        interval = (search_areas - block_sizes) // 2
        # Construct a template image and add 0 to the previous frame image
        mask_image_1 = np.zeros((height + interval * 2,  width + interval * 2))

        mask_image_1[:mask_image_1.shape[0] - interval*2 , :mask_image_1.shape[1] - interval*2] = np.array(reconstructed_image)

        mask_width, mask_height = mask_image_1.shape

        predict_image = np.zeros(reconstructed_image.shape)
        # print([[[0, 0] for j in range(height_num)] for i in range(width_num)])
        #     count = 0
        for i in range(height_num):
            for j in range(width_num):
                #             count += 1
                #         print(f'==================i:{i}=j:{j}==count:{count}=====================')
                temp_image = original_frame[i * block_sizes:(i + 1) * block_sizes, j * block_sizes:(j + 1) * block_sizes]
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
                            motion_vectors[i ][j][0], motion_vectors[i][j][1] = k, h

                            motion_vectors_for_draw[i][j][0], motion_vectors_for_draw[i][j][1], motion_vectors_for_draw[i][j][2], motion_vectors_for_draw[i][j][3] = \
                                i * search_areas + search_areas//2, j*search_areas + search_areas//2, \
                                     i*search_areas + k*block_sizes + search_areas//2,  j*search_areas + h*block_sizes + search_areas//2
                            predict_image[i * block_sizes:(i + 1) * block_sizes,
                            j * block_sizes:(j + 1) * block_sizes] = temp_mask
        #                         print(motion_vectors_for_draw[i*j+j])
        return np.array(predict_image), np.array(motion_vectors), np.array(motion_vectors_for_draw)

    def residual_compression(self, original_frame, predicted_frame):
        return np.subtract(original_frame, predicted_frame)




