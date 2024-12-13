import cv2
import time
import argparse
import numpy as np
from FM2S import FM2S


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-SS", "--sample_size", type=int, default=25)
    parser.add_argument("-EPI", "--epoch_per_image", type=int, default=150)
    parser.add_argument("-AMP", "--amplification_factor", type=int, default=2)
    parser.add_argument("-in_path", "--input_image_path", type=str, required=True)
    parser.add_argument("-out_path", "--output_image_path", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    raw = cv2.imread(args.input_image_path)
    denoised = []
    t0 = time.time()
    for i in range(raw.shape[2]):
        print(f"Denoising channel {i+1}/{raw.shape[2]}...")
        denoised_channel = FM2S(
            raw[:, :, i],
            args.sample_size,
            args.epoch_per_image,
            args.amplification_factor,
        )
        denoised.append(denoised_channel)
    denoised = np.stack(denoised, axis=2)
    t = time.time() - t0
    print(f"Time:{t}s")
    cv2.imwrite(args.output_image_path, denoised)


if __name__ == "__main__":
    args = get_args()
    main(args)
    print("Finished!")
