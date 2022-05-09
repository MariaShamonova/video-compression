import dataclasses
import math
import time

import numpy as np
from constants import MATRIX_QUANTIZATION


@dataclasses.dataclass
class Decoder:

    @staticmethod
    def _get_matrix_A(N):
        A = [[np.round(math.sqrt((1 if (i == 0) else 2) / N) * math.cos(((2 * j + 1) * i * math.pi) / (2 * N)), 3)
              for j in range(0, N)] for i in range(0, N)]
        return A


    def idct(self, Y, N):
        A = self._get_matrix_A(N)

        return np.array(A).transpose().dot(Y).dot(np.array(A))

    def dequantization(self, Y):
        return np.multiply(np.array(Y), MATRIX_QUANTIZATION)

    def inverse_zig_zag_transform(self, zigzag, N):
        blocks_8x8 = np.zeros((N, N))

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
            i = (N - len(slice) if index % 2 == 0 else N - 1)
            j = (N - len(slice) if index % 2 == 1 else N - 1)

            for ind in range(len(slice)):
                blocks_8x8[i][j] = int(zigzag[0])
                zigzag.pop(0)
                i = i + (1 if index % 2 == 0 else -1)
                j = j + (1 if index % 2 == 1 else -1)

        return blocks_8x8.astype(int)

    @staticmethod
    def entropy_decoder(bit_stream, codewars, N, shape):
        values = []
        bites = ''

        codewars = dict((v, k) for k, v in codewars.items())

        countValues = 0
        i = 0

        rows = int(shape[0] / 8)
        columns = int(shape[1] / 8)

        r = 0
        c = 0

        blocks = [[[] for i in range(columns)] for j in range(rows)]

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
                        values.extend([0 for z in range(int(N * N - len(values)))])

                        blocks[r][c] = values
                        r = (r + 1 if r < rows - 1 else 0)
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

        r = blocks.shape[0]

        for i in range(r):
            frame.append(np.concatenate((blocks[i]), axis=1))

        for i in range(0, r):
            if r > 1:
                frame.append(np.concatenate((frame[i]), axis=2))

        return np.array(frame[0])






