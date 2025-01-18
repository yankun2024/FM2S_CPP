import cv2
import argparse
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--ground_truth", type=str, required=True)
    parser.add_argument("-test", "--test_img", type=str, required=True)
    args = parser.parse_args()
    return args


def main(args):
    gt = cv2.imread(args.ground_truth)
    test = cv2.imread(args.test_img)
    psnr = peak_signal_noise_ratio(gt, test)
    ssim = structural_similarity(gt, test, channel_axis=2)
    print(f"PSNR:{psnr},SSIM:{ssim}")


if __name__ == "__main__":
    args = get_args()
    main(args)
