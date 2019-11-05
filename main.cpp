#include <cstdio>
#include <omp.h>
#include <opencv2/opencv.hpp>

/*
 * Laufzeit-Messungen für 2 4 6 8 10 12 14 16 Threads --> Daraus Bar-Chart machen
 * für OpenCV sowie für eigenen Algorithmus
 */

cv::Mat applyGrayscale(cv::Mat srcImage) {
    auto numThreads = omp_get_num_procs();
    auto destImage = cv::Mat(srcImage.rows, srcImage.cols, CV_8UC1);

    omp_set_num_threads(numThreads);

    #pragma omp parallel
    {
        for (int row = omp_get_thread_num(); row < srcImage.rows; row += omp_get_num_threads()) {

            for (int col = 0; col < srcImage.cols; col++) {
                auto srcPixel = srcImage.at<cv::Vec3b>(row, col);

                uchar r = srcPixel[2];
                uchar g = srcPixel[1];
                uchar b = srcPixel[0];


                // uchar destPixel = 0.21 * r + 0.72 * g + 0.07 * b; // luminosity formular
                uchar destPixel = 0.299 * r + 0.587 * g + 0.114 * b; // open cv formular

                destImage.at<uchar>(row, col) = destPixel;
            }
        }
    };

    return destImage;
}


cv::Mat applyHSV(cv::Mat srcImage) {
    auto numThreads = omp_get_num_procs();
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC3);

    omp_set_num_threads(numThreads);

    #pragma omp parallel
    {
        for (int row = omp_get_thread_num(); row < srcImage.rows; row += omp_get_num_threads()) {
            for (int col = 0; col < srcImage.cols; col++) {
                auto srcPixel = srcImage.at<cv::Vec3b>(row, col);

                double r = srcPixel[2] / 255.0;
                double g = srcPixel[1] / 255.0;
                double b = srcPixel[0] / 255.0;

                double h, s, v;

                double cMax = std::max(std::max(r, g), b);
                double cMin = std::min(std::min(r, g), b);
                double diff = cMax - cMin;

                if (cMax == cMin) {
                    h = 0;
                } else if (cMax == r) {
                    h = int(60 * ((g - b) / diff) + 360) % 360;
                } else if (cMax == g) {
                    h = int(60 * ((b - r) / diff) + 120) % 360;
                } else {
                    h = int(60 * ((r - g) / diff) + 240) % 360;
                }

                s = cMax == 0 ? 0 : diff / cMax;
                v = cMax;

                cv::Vec3b destPixel = cv::Vec3b(
                        uchar(h / 360.0 * 255.0),
                        uchar(s * 255.0),
                        uchar(v * 255.0)
                );

                destImage.at<cv::Vec3b>(row, col) = destPixel;
            }
        }
    };

    return destImage;
}

cv::Mat applyEmboss(cv::Mat srcImage) {
    return srcImage;
}


cv::Mat applyOpenCVGrayscale(const cv::Mat &srcImage) {
    cv::Mat destImage;
    cv::cvtColor(srcImage, destImage, cv::COLOR_BGR2GRAY);

    return destImage;
}

cv::Mat applyOpenCVHSV(const cv::Mat &srcImage) {
    cv::Mat destImage;
    cv::cvtColor(srcImage, destImage, cv::COLOR_BGR2HSV_FULL);

    return destImage;
}


cv::Mat applyOpenCVEmboss(const cv::Mat &srcImage) {
    float embossKernel[3][3] = {
            {0.0, -1.0, 0.0},
            {0.0, 0.0,  0.0},
            {0.0, 1.0,  0.0}
    };

    cv::Mat destImage;
    cv::Mat kernel = cv::Mat(3, 3, CV_32F, &embossKernel);
    cv::filter2D(srcImage, destImage, -1, kernel, cv::Point(-1, -1));

    return destImage + 128.0;
}

template<typename F>
void measureTime(F func, int iterations = 20) {
    func();
}

int main() {


    auto imagePath = "resources/images/dice.png";
    cv::Mat srcImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::Mat ownGrayscaleImage;
    cv::Mat cvGrayscaleImage;

    cv::Mat ownHSVImage;
    cv::Mat cvHSVImage;

    cv::Mat ownEmbossImage;
    cv::Mat cvEmbossImage;


    if (!srcImage.data) {
        printf("No srcImage data\n");

    } else {
        ownGrayscaleImage = applyGrayscale(srcImage);
        cvGrayscaleImage = applyOpenCVGrayscale(srcImage);

        ownHSVImage = applyHSV(srcImage);
        cvHSVImage = applyOpenCVHSV(srcImage);

        ownEmbossImage = applyEmboss(srcImage);
        cvEmbossImage = applyOpenCVEmboss(srcImage);


        imshow("Source Image", srcImage);
        // imshow("Own Grayscale", ownGrayscaleImage);
        // imshow("CV Grayscale", cvGrayscaleImage);
        // imshow("Difference Grayscale", abs(cvGrayscaleImage - ownGrayscaleImage));

        imshow("Own HSV", ownHSVImage);
        imshow("CV HSV", cvHSVImage);
        imshow("Difference HSV", abs(cvHSVImage - ownHSVImage));


        // imshow("Own Emboss", ownEmbossImage);
        // imshow("CV Emboss", cvEmbossImage);
        // imshow("Difference Grayscale", abs(cvEmbossImage - ownEmbossImage));
    }

    cv::waitKey(0);


    return 0;

}


