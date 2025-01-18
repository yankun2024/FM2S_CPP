#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "FM2S.h"

int main(int argc, char* argv[]) {
    torch::manual_seed(3407);

    int sample_size = 25;
    int epoch_per_image = 150;
    int amplification_factor = 2;
    std::string input_image_path;
    std::string output_image_path;

    // get_args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-SS" || arg == "--sample_size") {
            sample_size = std::stoi(argv[++i]);
        }
        else if (arg == "-EPI" || arg == "--epoch_per_image") {
            epoch_per_image = std::stoi(argv[++i]);
        }
        else if (arg == "-AMP" || arg == "--amplification_factor") {
            amplification_factor = std::stoi(argv[++i]);
        }
        else if (arg == "-i" || arg == "--input_image_path") {
            input_image_path = argv[++i];
        }
        else if (arg == "-o" || arg == "--output_image_path") {
            output_image_path = argv[++i];
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    cv::Mat raw = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);
    if (raw.empty()) {
        std::cerr << "Error: Could not load the image.\n";
        return -1;
    }

    try {
        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat denoised = FM2S(raw, sample_size, epoch_per_image, amplification_factor);
        std::chrono::duration<double> t = std::chrono::high_resolution_clock::now() - t0;
        std::cout << "Time: " << t.count() << "s" << std::endl;
        cv::imwrite(output_image_path, denoised);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }

    std::cout << "Finished!" << std::endl;
    return 0;
}
