#include <cstdio>
#include <omp.h>
#include <opencv2/opencv.hpp>

cv::Mat applyGrayscale(cv::Mat image) {
    return image;
}

cv::Mat applyHSV(cv::Mat image) {
    return image;
}

cv::Mat applyEmboss(cv::Mat image) {
    return image;
}


cv::Mat applyOpenCVGrayscale(const cv::Mat &image) {
    cv::Mat convertedImage;

    cv::cvtColor(image, convertedImage, cv::COLOR_BGR2GRAY);

    return convertedImage;
}

cv::Mat applyOpenCVHSV(const cv::Mat &image) {
    cv::Mat convertedImage;

    cv::cvtColor(image, convertedImage, cv::COLOR_BGR2HSV);

    return convertedImage;
}


cv::Mat applyOpenCVEmboss(const cv::Mat &image) {
    auto srcImage = applyOpenCVGrayscale(image);
    //const auto &srcImage = image;

    float embossKernel1[3][3] = {
            {-2.0, -1.0, 0.0},
            {-1.0, 1.0, 1.0},
            {0.0, 1.0, 2.0}
    };

    float embossKernel2[3][3] = {
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0},
            {0.0, -1.0, 0.0}
    };

    float embossKernel3[3][3] = {
            {-1.0, -1.0, 0.0},
            {-1.0, 0.0, 1.0},
            {0.0, 1.0, 1.0}
    };


    float embossKernel4[3][3] = {
            {1.0, 0.0, 1.0},
            {0.0,  -4.0, 0.0},
            {1.0, 0.0, 1.0}
    };


    cv::Mat convertedImage;
    cv::Mat kernel = cv::Mat(3, 3, CV_32F, &embossKernel1);


    cv::filter2D(srcImage, convertedImage, -1, kernel, cv::Point(-1, -1));

    return convertedImage;
}


int main() {
#pragma omp parallel default(none)
    for (int i = 0; i < 8; i++) {
        printf("Hallo iteration %d from thread %d / %d!\n", i, omp_get_thread_num(), omp_get_num_threads());
    }

    auto imagePath = "resources/images/dice.png";
    auto image = cv::imread(imagePath);

    if (!image.data) {
        printf("No image data\n");

    } else {
        namedWindow("Display Image", cv::WINDOW_AUTOSIZE);

        // image = applyOpenCVGrayscale(image);
        // image = applyOpenCVHSV(image);
        //image = applyOpenCVEmboss(image);

        imshow("Display Image", image);
    }

    cv::waitKey(0);


    return 0;
}


