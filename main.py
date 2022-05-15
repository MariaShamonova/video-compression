# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt
from reshapeFrame import reshapeFrame
from encoder import Encoder
from decoder import Decoder

p_frame_thresh = 300000
N = 8


def degreeOfSimilarity(block1, block2):
    # Евклидовая норма разницы двух матриц
    diff = block1 - block2
    np.linalg.norm(diff, ord=None, axis=None, keepdims=False)


def splitIntoDisjointBlocks(block):
    result = block.reshape(3, 4, 4)
    return result


def searchForKeyframes(curr_frame, prev_frame):
    diff = cv2.absdiff(curr_frame, prev_frame)
    non_zero_count = np.count_nonzero(diff)
    if (non_zero_count > p_frame_thresh):
        print("Got P-Frame")
        prev_frame = curr_frame
        return "P"
    return "I"
    print('search key frames')


def create_rec_frame(reconstructed_frame):
    with open('reconstructed.txt', "wb") as f:
        pickle.dump(reconstructed_frame, f)


def read_rec_frame():
    with open('reconstructed.txt', 'rb') as f:
        b = pickle.load(f)
    return b


def draw_motion_vectors(image, motion_vector_draw):
    print('draw_motion_vectors')

    color = (0, 255, 0)
    thickness = 2
    height, width, index = image.shape
    width_num = width // block_sizes
    height_num = height // block_sizes

    for i in range(height_num):
        for j in range(width_num):

            # print(motion_vector_draw[i*j+j])
            start_point = (int(motion_vector_draw[i][j][0]), int(motion_vector_draw[i][j][1]))
            end_point = (int(motion_vector_draw[i][j][2]), int(motion_vector_draw[i][j][3]))
            if start_point != end_point:
                image = cv2.arrowedLine(image, start_point, end_point,
                                    color, thickness)

    print(image.shape)

    return image

def write_video_file(frames):
    height, width = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter('project.avi', 0, 1, size)

    # for i in range(len(frames)):
    #     image = (frames[i] * 255).astype(np.uint8)
    #     out.write(image)
    # out.release()


if __name__ == "__main__":
    cap = cv2.VideoCapture('video.MOV')
    encoder = Encoder()
    decoder = Decoder()

    i = 0
    frames = []

    ret, frame = cap.read()

    reconstructed_frames = []

    # while(cap.isOpened()):
    while(i < 5):

        ret, frame = cap.read()
        frame_y = encoder.transform_rgb_to_y(frame)
        if i == 0 or i == 4:
            if i % 5 == 0:
                # bit_stream, dict_Haffman, frame_y = encoder.encode_I_frame(frame=frame)
                print(i)
                quantinization_coeff = encoder.encode_I_frame(frame=frame)
            else:

                # bit_stream, dict_Haffman, frame_y = encoder.encode_B_frame(frame=frame, reconstructed_frame=reconstructed_frames[i - 1])
                quantinization_coeff, motion_vectors,  predict_image, residual_frame = encoder.encode_B_frame(frame=frame, reconstructed_frame=reconstructed_frames[len(reconstructed_frames) - 1])

            # inverse_transformed_frame = decoder.decode(bit_stream, dict_Haffman, frame_y.shape)
            inverse_transformed_frame = decoder.decode(quantinization_coeff)

            if i % 5 == 0:
                reconstructed_frames.append(inverse_transformed_frame)
            else:
                #Прибавить предыдущий реконструированный кадр
                rec, inv_predicted_image, inv_residual_frame = decoder.decode_B_frame(inverse_transformed_frame, reconstructed_frames[len(reconstructed_frames) - 1], motion_vectors)
                reconstructed_frames.append(rec)
                fig = plt.figure()
                ax = fig.add_subplot(3, 2, 1)
                ax.set_title('Original')
                ax.imshow(reconstructed_frames[len(reconstructed_frames) - 1], cmap=plt.get_cmap(name='gray'))
                ax2 = fig.add_subplot(3, 2, 2)
                ax2.set_title('Decoded image')
                ax2.imshow(rec, cmap=plt.get_cmap(name='gray'))
                ax3 = fig.add_subplot(3, 2, 3)
                ax3.set_title('predict_image')
                ax3.imshow(predict_image, cmap=plt.get_cmap(name='gray'))
                ax4 = fig.add_subplot(3, 2, 4)
                ax4.set_title('residual_frame')
                ax4.imshow( residual_frame, cmap=plt.get_cmap(name='gray'))

                ax5 = fig.add_subplot(3, 2, 5)
                ax5.set_title('inv_predicted_image')
                ax5.imshow(inv_predicted_image, cmap=plt.get_cmap(name='gray'))
                ax6 = fig.add_subplot(3, 2, 6)
                ax6.set_title('inv_residual_frame')
                ax6.imshow( inv_residual_frame, cmap=plt.get_cmap(name='gray'))
                plt.show()
                print(i)
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 2, 1)
        # ax.imshow(frame_y, cmap=plt.get_cmap(name='gray'))
        # ax2 = fig.add_subplot(1, 2, 2)
        # ax2.imshow(reconstructed_frames[i], cmap=plt.get_cmap(name='gray'))
        # plt.show()

        if ret == False:
            break

        frames.append(frame)

        i += 1

    cap.release()

    # write_video_file(reconstructed_frames)
    # frame = cv2.imread('frame_1.jpeg', cv2.COLOR_RGB2BGR)
    # bit_stream, codewars, frame_y = encode(frames[0])

    # reconstructed_frame = decode(bit_stream, codewars, frame_y.shape)
    # create_rec_frame(reconstructed_frame)
    # reconstructed_frame = read_rec_frame()
    # enc = Encoder()
    # height, width, index = frames[5].shape
    # block_sizes = 16
    # search_areas = 64
    #
    # frame_y = enc.transform_rgb_to_y(frames[5])
    #
    # predict_image, motion_vectors, motion_vectors_for_draw = enc.motion_estimation(reconstructed_frame, frame_y, width, height, block_sizes, search_areas)
    # image = draw_motion_vectors(frames[5], motion_vectors_for_draw)
    # residual_frame = enc.residual_compression(frame_y, predict_image)
    # bit_stream, dict_Haffman, frame_y = encode(residual_frame)
    # reconstructed_frame = decode(bit_stream, dict_Haffman, frame_y.shape)


    # height, width = frame_y.shape
    # div = (2160, 3840)
    # upsampled_image = cv2.resize(decode_frame_y, div, interpolation=cv2.INTER_CUBIC)


    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(reconstructed_frame, cmap=plt.get_cmap(name='gray'))
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.imshow(residual_frame, cmap=plt.get_cmap(name='gray'))
    # plt.show()

    # cv2.namedWindow('displaymywindows', cv2.WINDOW_NORMAL)
    # cv2.imshow('displaymywindows', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()







