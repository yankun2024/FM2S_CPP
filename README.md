# FM2S: Self-Supervised Fluorescence Micrograph Denoising With Single Noisy Image

- <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.09613-red?logo=arxiv" height="14" />  [FM2S: Self-Supervised Fluorescence Microscopy Denoising With Single Noisy Image](https://arxiv.org/abs/2412.10031).

## This is a C++ variant designed for easier integration into pure C projects.
## Original Author

This project is based on the work of the original author:  
[@Danielement321](https://github.com/Danielement321/FM2S)

## Getting Start 

To compile the project, the following two dependency libraries are required:
1.To https://pytorch.org/ Download libtorch
2.To https://opencv.org/ Download opencv
If you are on the Windows platform, you can easily download the corresponding ready-to-use version above.

Refer to the example below to modify your CMakeLists.txt.

	cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
	project(FM2S_CPP)

	set(OpenCV_PREFIX_PATH "./opencv/build") # fix your opencv path
	list(APPEND CMAKE_PREFIX_PATH ${OpenCV_PREFIX_PATH})
	find_package(OpenCV REQUIRED)

	set(CMAKE_PREFIX_PATH "./libtorch") # fix your libtorch path
	find_package(Torch REQUIRED)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

	include_directories(${OpenCV_INCLUDE_DIRS})
	add_executable(FM2S_CPP main.cpp FM2S.h)
	target_link_libraries(FM2S_CPP "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

	set_property(TARGET FM2S_CPP PROPERTY CXX_STANDARD 17)

## Build

	mkdir build
	cd build
	cmake ..
	cmake --build .

If your system has not been configured with the NVIDIA development environment in detail before, please copy all the dynamic libraries from the 'libtorch\lib' directory to the same directory as the executable.

## Denoising

This project aims to integrate the algorithm into other engineering projects and provides only a test case.
If a image is in grayscale, use the following command:

	./FM2S_CPP.exe -in_path samples/grayscale/noisy.png -out_path g_out.png

Other modifications you may need can be referenced from the original Python version.

## Others
This code repository may be merged or deleted at any time as per the original author's requirements.
