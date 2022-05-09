import dataclasses
import math

import numpy as np
from constants import MATRIX_QUANTIZATION
from repository import get_probabilities, HuffmanTree


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







