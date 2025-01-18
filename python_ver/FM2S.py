import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from scipy import ndimage as nd

torch.manual_seed(3407)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class network(nn.Module):
    def __init__(self, amp):
        super(network, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=1e-3)
        self.conv1 = nn.Conv2d(amp, 24, 3, padding=1)
        self.conv2 = nn.Conv2d(24, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, amp, 3, padding=1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


# Arbitrary MPG generator
def regional_add_noise(x, gaussian_level, poisson_level):
    noisy = torch.poisson(x * poisson_level) / poisson_level
    noisy = noisy + torch.normal(0, gaussian_level / 255, x.shape).to(device)
    noisy = torch.clamp(noisy, 0, 1)
    return noisy


def noise_addition(img, stride=75):
    noisy_img = img.clone()
    # Region-Wise Noise Addition
    for h in range(0, noisy_img.shape[2], stride):
        for w in range(0, noisy_img.shape[3], stride):
            region = noisy_img[0, :, h : h + stride, w : w + stride]
            noise_idx = torch.mean(region).clamp(0.01, 0.20)
            noisy_img[0, :, h : h + stride, w : w + stride] = regional_add_noise(
                region,
                gaussian_level=200 * noise_idx,
                poisson_level=30 / noise_idx,
            )

    # Overall Noise Addition
    noisy_img = torch.poisson(noisy_img * 60) / 60
    noisy_img = torch.clamp(noisy_img, 0, 1)

    return noisy_img


def FM2S(raw_img, SS, EPI, amp):
    raw_img = raw_img / 255  # Image normalization
    clean_img = nd.median_filter(raw_img, 3)  # Median filter
    clean_img = torch.tensor(clean_img, dtype=torch.float32, device=device)
    raw_img = torch.tensor(raw_img, dtype=torch.float32, device=device)

    # Channel Amplification
    clean_img = clean_img.unsqueeze(0).repeat(1, amp, 1, 1)
    raw_img = raw_img.unsqueeze(0).repeat(1, amp, 1, 1)

    model = network(amp).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.25)

    for sample in trange(SS):
        # Adds adaptive MPG to filtered image
        noisy_img = noise_addition(clean_img)

        for epoch in range(EPI):
            pred = model(noisy_img)

            # Filtered image also serves as the training target
            loss = criterion(pred, clean_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    with torch.no_grad():
        denoised_img = model(raw_img)

    denoised_img = torch.clamp(denoised_img, 0, 1) * 255
    denoised_img = torch.mean(denoised_img, dim=1).squeeze()  # Image average
    denoised_img = denoised_img.cpu().int().numpy()
    return denoised_img
