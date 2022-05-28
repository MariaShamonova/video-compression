# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import pickle

import numpy as np
from matplotlib import pyplot as plt

from encoder import Encoder
from decoder import Decoder
from frame import Frame
from repository import concat_blocks

METHOD = 0


def create_rec_frame(reconstructed_frame):
    with open("reconstructed.txt", "wb") as f:
        pickle.dump(reconstructed_frame, f)


def read_rec_frame():
    with open("reconstructed.txt", "rb") as f:
        b = pickle.load(f)
    return b


def check_key_frame(index):
    return index % 5 == 0



if __name__ == "__main__":
    cap = cv2.VideoCapture("video2.mp4")
    framespersecond = int(cap.get(cv2.CAP_PROP_FPS))
    print("number of frames in this video is ", framespersecond)

    encoder = Encoder()
    decoder = Decoder()

    i = 0
    frames = []

    # ret, frame = cap.read()

    reconstructed_frames = []

    while i < 5:

        ret, frame = cap.read()
        HEIGHT, WIDTH, num_channels = frame.shape

        frame = Frame(
            frame=frame, is_key_frame=check_key_frame(i), width=WIDTH, height=HEIGHT
        )




        if i == 0 or i == 1:
            if i % 5 == 0:
                # bit_stream, dict_Haffman, frame_y = encoder.encode_I_frame(frame=frame)
                encoded_frame = encoder.encode_I_frame(frame=frame, method=METHOD)
            else:
                # bit_stream, dict_Haffman, frame_y = encoder.encode_B_frame(frame=frame, reconstructed_frame=reconstructed_frames[i - 1])
                encoded_frame = encoder.encode_B_frame(
                    frame=frame,
                    reconstructed_frame=reconstructed_frames[
                        len(reconstructed_frames) - 1
                    ],
                    method=METHOD,
                )
            # temp_lum = concat_blocks(encoded_frame.channels.luminosity)
            # temp_cr = concat_blocks(encoded_frame.channels.chromaticCr)
            # temp_cb = concat_blocks(encoded_frame.channels.chromaticCb)
            # temp_channels = Channels(luminosity=temp_lum, chromaticCr=temp_cr, chromaticCb=temp_cb, is_encoded=False)
            # temp_frame = Frame(channels=temp_channels, is_key_frame=True, width=WIDTH, height=HEIGHT)
            # temp_frame.show_frame()

            # create_rec_frame(encoded_channels)
            # encoded_channels = read_rec_frame()

            # inverse_transformed_frame = decoder.decode(bit_stream, dict_Haffman, frame_y.shape)
            dequantized_frame = decoder.decode(encoded_frame, method=METHOD)

            if i % 5 == 0:
                reconstructed_frames.append(dequantized_frame)
                # cv2.imshow('deq', np.round(dequantized_frame.channels.luminosity, 0))
                # cv2.waitKey(0)
                # imgplot = plt.imshow(dequantized_frame.channels.luminosity, cmap='gray')
                # plt.show()
                # dequantized_frame.show_frame()
            else:
                # Прибавить предыдущий реконструированный кадр
                reconstructed_frame = decoder.decode_B_frame(
                    dequantized_frame,
                    reconstructed_frames[len(reconstructed_frames) - 1],
                    method=METHOD,
                )
                reconstructed_frames.append(reconstructed_frame)
                # fig = plt.figure()
                # ax = fig.add_subplot(3, 2, 1)
                # ax.set_title('Original')
                # ax.imshow(reconstructed_frames[len(reconstructed_frames) - 1], cmap=plt.get_cmap(name='gray'))
                # ax2 = fig.add_subplot(3, 2, 2)
                # ax2.set_title('Decoded image')
                # ax2.imshow(rec, cmap=plt.get_cmap(name='gray'))
                # ax3 = fig.add_subplot(3, 2, 3)
                # ax3.set_title('predict_image')
                # ax3.imshow(predict_image, cmap=plt.get_cmap(name='gray'))
                # ax4 = fig.add_subplot(3, 2, 4)
                # ax4.set_title('residual_frame')
                # ax4.imshow( residual_frame, cmap=plt.get_cmap(name='gray'))
                #
                # ax5 = fig.add_subplot(3, 2, 5)
                # ax5.set_title('inv_predicted_image')
                # ax5.imshow(inv_predicted_image, cmap=plt.get_cmap(name='gray'))
                # ax6 = fig.add_subplot(3, 2, 6)
                # ax6.set_title('inv_residual_frame')
                # ax6.imshow( inv_residual_frame, cmap=plt.get_cmap(name='gray'))
                # plt.show()
                print(i)

                # reconstructed_frame.show_frame()

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 2, 1)
        # ax.imshow(frame_y, cmap=plt.get_cmap(name='gray'))
        # ax2 = fig.add_subplot(1, 2, 2)
        # ax2.imshow(reconstructed_frames[i], cmap=plt.get_cmap(name='gray'))
        # plt.show()

        if ret == False:
            break



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
