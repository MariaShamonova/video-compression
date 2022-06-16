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
from repository import compute_mse, compute_psnr, compression_ratio

METHOD = 1


def create_file(bit_stream):
    with open("reconstructed.bin", "wb") as f:
        pickle.dump(bit_stream, f)


def read_rec_frame():
    with open("reconstructed.bin", "rb") as f:
        b = pickle.load(f)
    return b


def check_key_frame(index):
    return index % 5 == 0


if __name__ == "__main__":
    cap = cv2.VideoCapture("video3.mp4")
    framespersecond = int(cap.get(cv2.CAP_PROP_FPS))
    print("number of frames in this video is ", framespersecond)

    encoder = Encoder()
    decoder = Decoder()

    i = 0
    frames = []

    reconstructed_frames = []
    main_bit_stream = ""
    while i < 5:

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        original_frame = frame
        HEIGHT, WIDTH, num_channels = frame.shape
        is_key_frame = check_key_frame(i)
        frame = Frame(
            frame=frame,
            is_key_frame=is_key_frame,
            width=WIDTH,
            height=HEIGHT,
            method=METHOD,
        )
        if i == 0 or i == 4:
            if i % 5 == 0:
                bit_stream, dictionary = encoder.encode_I_frame(
                    frame=frame, method=METHOD
                )
            else:
                bit_stream, dictionary = encoder.encode_B_frame(
                    frame=frame,
                    reconstructed_frame=reconstructed_frames[
                        len(reconstructed_frames) - 1
                    ],
                    method=METHOD,
                )
            main_bit_stream += bit_stream
            dequantized_frame = decoder.decode(
                bit_stream=bit_stream,
                dictionary=dictionary,
                is_key_frame=is_key_frame,
                method=METHOD,
                width=WIDTH,
                height=HEIGHT,
            )

            if i % 5 == 0:
                reconstructed_frames.append(dequantized_frame)
                mse_error = compute_mse(original_frame, dequantized_frame.build_frame())
                psnr_error = compute_psnr(
                    original_frame, dequantized_frame.build_frame()
                )

                print("MSE ERROR I-frame: ", mse_error)
                print("PSNR ERROR I-frame: ", psnr_error)

            else:

                reconstructed_frame = decoder.decode_B_frame(
                    dequantized_frame,
                    reconstructed_frames[len(reconstructed_frames) - 1],
                    method=METHOD,
                )
                mse_error = compute_mse(original_frame, dequantized_frame.build_frame())
                psnr_error = compute_psnr(
                    original_frame, dequantized_frame.build_frame()
                )

                print("MSE ERROR B-frame: ", mse_error)
                print("PSNR ERROR B-frame: ", psnr_error)

            if ret == False:
                break

            if cv2.waitKey(1) == ord("q"):
                break

        i += 1

    cap.release()
    create_file(main_bit_stream)
    cv2.destroyAllWindows()
