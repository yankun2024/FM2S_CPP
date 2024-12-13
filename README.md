# FM2S: Self-Supervised Fluorescence Micrograph Denoising With Single Noisy Image
## Getting Start
It is recommanded to use virtual environment.

	conda create -n FM2S python=3.9
	conda activate FM2S
	pip install -r requirements.txt

## Denoising
If a image is in grayscale, use the following command

	python main.py -in_path samples/grayscale/noisy.png -out_path g_out.png

For a color image, use the following command

	python main4D.py -in_path samples/color/noisy.png -out_path c_out.png

## Hyperparameters
A set of hyperparameters can be modified, including Sample Size (SS), Epoch Per Image (EPI), and Amplification Factor (AMP).

	python main.py -in_path samples/grayscale/noisy.png -out_path g_out.png -SS 75 -EPI 150 -AMP 2

## Evaluation
PSNR and SSIM are used for denoising evaluation. To compute PSNR and SSIM for a denoised image, use the following command

	python eva.py -gt samples/grayscale/gt.png -test g_out.png