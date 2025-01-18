#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

torch::Device device(torch::cuda::is_available() ? torch::kCUDA: torch::kCPU);

struct Network : torch::nn::Module {
    torch::nn::LeakyReLU act;
    torch::nn::Conv2d conv1, conv2, conv3;

    explicit Network(int amp)
        : act(torch::nn::LeakyReLUOptions().negative_slope(1e-3)),
          conv1(torch::nn::Conv2dOptions(amp, 24, 3).padding(1)),
          conv2(torch::nn::Conv2dOptions(24, 12, 3).padding(1)),
          conv3(torch::nn::Conv2dOptions(12, amp, 3).padding(1)) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = act(conv1(x));
        x = act(conv2(x));
        x = conv3(x);
        return x;
    }
};

// Arbitrary MPG generator
torch::Tensor regional_add_noise(torch::Tensor x, float gaussian_level, float poisson_level) {
    auto noisy = torch::poisson(x * poisson_level) / poisson_level;
    noisy = noisy + torch::normal(0.0, gaussian_level / 255.0, x.sizes()).to(device);;
    return torch::clamp(noisy, 0.0, 1.0);
}

torch::Tensor noise_addition(torch::Tensor img, int stride = 75) {
    auto noisy_img = img.clone();
    auto height = noisy_img.size(2);
    auto width = noisy_img.size(3);
    // Region-Wise Noise Addition
    for (int h = 0; h < height; h += stride) {
        for (int w = 0; w < width; w += stride) {
            auto region = noisy_img.slice(2, h, std::min<int>(h + stride, height))
                                   .slice(3, w, std::min<int>(w + stride, width));
            float noise_idx = torch::mean(region).clamp(0.01, 0.20).item<float>();
            noisy_img.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(),
                torch::indexing::Slice(h, std::min<int>(h + stride, height)),
                torch::indexing::Slice(w, std::min<int>(w + stride, width))
            }) = regional_add_noise(region, 200 * noise_idx, 30 / noise_idx);
        }
    }

    // Overall Noise Addition
    noisy_img = torch::poisson(noisy_img * 60) / 60;
    return torch::clamp(noisy_img, 0.0, 1.0);
}

cv::Mat FM2S(const cv::Mat& raw_img, int SS, int EPI, int amp) {
    //cv::Mat raw_img_x = raw_img.clone() /255;
    cv::Mat clean_img_cv;
    // 执行中值滤波，内核大小为 3x3
    cv::medianBlur(raw_img, clean_img_cv, 3);
    // Image normalization And to Tensor
    torch::Tensor clean_img = torch::from_blob(
        clean_img_cv.data, {raw_img.rows, raw_img.cols}, torch::kUInt8
    ).to(torch::kFloat32).to(device)/255.0;
    torch::Tensor raw_img_tensor = torch::from_blob(
        raw_img.data, {raw_img.rows, raw_img.cols}, torch::kUInt8
    ).to(torch::kFloat32).to(device)/255.0;
   // auto clean_img = raw_img_tensor.clone().to(device);

    // Channel amplification
    clean_img = clean_img.unsqueeze(0).repeat({1, amp, 1, 1});
    raw_img_tensor = raw_img_tensor.unsqueeze(0).repeat({1, amp, 1, 1});

    auto model = std::make_shared<Network>(amp);
    model->to(device);
    model->train();

    auto criterion = torch::nn::MSELoss();
    //auto L1_criterion = torch::nn::L1Loss();
    torch::optim::Adam optimizer(model->parameters());
    torch::optim::StepLR scheduler(optimizer, 1500, 0.25);

    for (int sample = 0; sample < SS; ++sample) {
        // Adds adaptive MPG to filtered image
        auto noisy_img = noise_addition(clean_img);
        //float lambda_ = 0.15;
        for (int epoch = 0; epoch < EPI; ++epoch) {
            auto pred = model->forward(noisy_img);

            // Filtered image also serves as the training target
            auto loss = criterion(pred, clean_img); //+ lambda_ * L1_criterion(pred, noisy_img);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            scheduler.step();

            // No tqdm so we manual print it
            int progress = static_cast<int>((static_cast<float>(epoch + 1) / EPI) * 100);
            std::cout << "\rSample " << sample + 1 << "/" << SS << " | Epoch " << epoch + 1 << "/" << EPI 
                      << " | Progress: [" << std::string(progress / 2, '#') 
                      << std::string(50 - progress / 2, '-') << "] " 
                      << progress << "%   ";
            std::cout.flush();
        }
    }

    // no_grad
    model->eval();
    torch::NoGradGuard no_grad;
    auto denoised_img = model->forward(raw_img_tensor);
    
    denoised_img = torch::clamp(denoised_img, 0.0, 1.0) * 255;
    denoised_img = denoised_img.mean(1).squeeze().to(torch::kUInt8).cpu(); // Image average And To cpu
    cv::Mat output(raw_img.rows, raw_img.cols, CV_8UC1, denoised_img.data_ptr());
    return output.clone();
}