#include <cstdio>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <chrono>

/*
 * TODO: Laufzeit-Messungen für 2 4 6 8 10 12 14 16 Threads --> Daraus Bar-Chart machen für OpenCV sowie für eigenen Algorithmus
 * TODO: Bei Emboss könnte der Alpha-Channel beachtet werden.
 *
 */

cv::Mat applyGrayscaleOuter(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    auto destImage = cv::Mat(srcImage.rows, srcImage.cols, CV_8UC1);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for default(none) shared(srcImage, destImage)
    for (int row = 0; row < srcImage.rows; row++) {
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

    return destImage;
}

cv::Mat applyGrayscaleInner(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    auto destImage = cv::Mat(srcImage.rows, srcImage.cols, CV_8UC1);
    omp_set_num_threads(numThreads);

    for (int row = 0; row < srcImage.rows; row++) {
        #pragma omp parallel for default(none) shared(srcImage, destImage, row)
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

    return destImage;
}

cv::Mat applyGrayscaleBoth(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    auto destImage = cv::Mat(srcImage.rows, srcImage.cols, CV_8UC1);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for default(none) shared(srcImage, destImage)
    for (int row = 0; row < srcImage.rows; row++) {
        #pragma omp parallel for default(none) shared(srcImage, destImage, row)
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

    return destImage;
}

cv::Mat applyHsvOuter(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC3);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for default(none) shared(srcImage, destImage)
    for (int row = 0; row < srcImage.rows; row++) {
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

            s = cMax == 0 ? 0 : (diff / cMax);
            v = cMax;

            cv::Vec3b destPixel = cv::Vec3b(
                    uchar(h / 360.0 * 180.0),
                    uchar(s * 255.0),
                    uchar(v * 255.0)
            );

            destImage.at<cv::Vec3b>(row, col) = destPixel;
        }
    }

    return destImage;
}

cv::Mat applyHsvInner(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC3);
    omp_set_num_threads(numThreads);

    for (int row = 0; row < srcImage.rows; row++) {
        #pragma omp parallel for default(none) shared(srcImage, destImage, row)
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

            s = cMax == 0 ? 0 : (diff / cMax);
            v = cMax;

            cv::Vec3b destPixel = cv::Vec3b(
                    uchar(h / 360.0 * 180.0),
                    uchar(s * 255.0),
                    uchar(v * 255.0)
            );

            destImage.at<cv::Vec3b>(row, col) = destPixel;
        }
    }

    return destImage;
}

cv::Mat applyHsvBoth(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC3);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for default(none) shared(srcImage, destImage)
    for (int row = 0; row < srcImage.rows; row++) {
        #pragma omp parallel for default(none) shared(srcImage, destImage, row)
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

            s = cMax == 0 ? 0 : (diff / cMax);
            v = cMax;

            cv::Vec3b destPixel = cv::Vec3b(
                    uchar(h / 360.0 * 180.0),
                    uchar(s * 255.0),
                    uchar(v * 255.0)
            );

            destImage.at<cv::Vec3b>(row, col) = destPixel;
        }
    }

    return destImage;
}

cv::Mat applyEmbossOuter(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC1);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for default(none) shared(srcImage, destImage)
    for (int row = 0; row < srcImage.rows; row++) {
        for (int col = 0; col < srcImage.cols; col++) {
            int diffR, diffG, diffB, diff, gray;
            auto srcPixel = srcImage.at<cv::Vec3b>(row, col);

            if (row == 0 || col == 0) {
                diffR = srcPixel[2];
                diffG = srcPixel[1];
                diffB = srcPixel[0];
            } else {
                auto upperLeftPixel = srcImage.at<cv::Vec3b>(row - 1, col - 1);
                diffR = std::abs(srcPixel[2] - upperLeftPixel[2]);
                diffG = std::abs(srcPixel[1] - upperLeftPixel[1]);
                diffB = std::abs(srcPixel[0] - upperLeftPixel[0]);
            }

            diff = std::max(diffR, std::max(diffG, diffB));

            gray = 128 + diff;
            gray = gray > 255 ? 255 : gray;
            gray = gray < 0 ? 0 : gray;

            destImage.at<uchar>(row, col) = gray;
        }
    }

    return destImage;
}

cv::Mat applyEmbossInner(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC1);
    omp_set_num_threads(numThreads);

    for (int row = 0; row < srcImage.rows; row++) {
        #pragma omp parallel for default(none) shared(srcImage, destImage, row)
        for (int col = 0; col < srcImage.cols; col++) {
            int diffR, diffG, diffB, diff, gray;
            auto srcPixel = srcImage.at<cv::Vec3b>(row, col);

            if (row == 0 || col == 0) {
                diffR = srcPixel[2];
                diffG = srcPixel[1];
                diffB = srcPixel[0];
            } else {
                auto upperLeftPixel = srcImage.at<cv::Vec3b>(row - 1, col - 1);
                diffR = std::abs(srcPixel[2] - upperLeftPixel[2]);
                diffG = std::abs(srcPixel[1] - upperLeftPixel[1]);
                diffB = std::abs(srcPixel[0] - upperLeftPixel[0]);
            }

            diff = std::max(diffR, std::max(diffG, diffB));

            gray = 128 + diff;
            gray = gray > 255 ? 255 : gray;
            gray = gray < 0 ? 0 : gray;

            destImage.at<uchar>(row, col) = gray;
        }
    }

    return destImage;
}

cv::Mat applyEmbossBoth(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC1);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for default(none) shared(srcImage, destImage)
    for (int row = 0; row < srcImage.rows; row++) {
        #pragma omp parallel for default(none) shared(srcImage, destImage, row)
        for (int col = 0; col < srcImage.cols; col++) {
            int diffR, diffG, diffB, diff, gray;
            auto srcPixel = srcImage.at<cv::Vec3b>(row, col);

            if (row == 0 || col == 0) {
                diffR = srcPixel[2];
                diffG = srcPixel[1];
                diffB = srcPixel[0];
            } else {
                auto upperLeftPixel = srcImage.at<cv::Vec3b>(row - 1, col - 1);
                diffR = std::abs(srcPixel[2] - upperLeftPixel[2]);
                diffG = std::abs(srcPixel[1] - upperLeftPixel[1]);
                diffB = std::abs(srcPixel[0] - upperLeftPixel[0]);
            }

            diff = std::max(diffR, std::max(diffG, diffB));

            gray = 128 + diff;
            gray = gray > 255 ? 255 : gray;
            gray = gray < 0 ? 0 : gray;

            destImage.at<uchar>(row, col) = gray;
        }
    }

    return destImage;
}

cv::Mat applyOpenCvGrayscale(const cv::Mat &srcImage) {
    cv::Mat destImage;
    cv::cvtColor(srcImage, destImage, cv::COLOR_BGR2GRAY);

    return destImage;
}

cv::Mat applyOpenCvHsv(const cv::Mat &srcImage) {
    cv::Mat destImage;
    cv::cvtColor(srcImage, destImage, cv::COLOR_BGR2HSV);
    return destImage;
}

template<typename F>
void measureTime(char *title, F func, int iterations = 20, int maxThreadCount = 16) {
    auto *diffs = new double[maxThreadCount];

    for (int j = 0; j < 16; j++) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            func(j + 1);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        diffs[j] = ((double) std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() /
                    iterations) /
                   1000.0;
    }


    std::cout << title << std::endl;
    for (int k = 0; k < maxThreadCount; k++) {
        std::cout << "(" << (k + 1) << "," << diffs[k] << ")";
    }

    std::cout << std::endl << std::endl;
}

int main() {
    auto imagePath = "resources/images/dice.png";
    cv::Mat srcImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::Mat ownGrayscaleImage;
    cv::Mat cvGrayscaleImage;

    cv::Mat ownHSVImage;
    cv::Mat cvHSVImage;

    cv::Mat ownEmbossImage;

    if (!srcImage.data) {
        printf("No srcImage data\n");

    } else {
        /* ---------- Grayscale ---------- */

        measureTime((char *) "Own Grayscale (Outer)", [&](int numThreads) {
            ownGrayscaleImage = applyGrayscaleOuter(srcImage, numThreads);
        });
        measureTime((char *) "Own Grayscale (Inner)", [&](int numThreads) {
            ownGrayscaleImage = applyGrayscaleInner(srcImage, numThreads);
        });
        measureTime((char *) "Own Grayscale (Both)", [&](int numThreads) {
            ownGrayscaleImage = applyGrayscaleBoth(srcImage, numThreads);
        });
        measureTime((char *) "OpenCV Grayscale", [&](int numThreads) {
            cvGrayscaleImage = applyOpenCvGrayscale(srcImage);
        });

        /* ---------- HSV ---------- */

        measureTime((char *) "Own HSV (Outer)", [&](int numThreads) {
            ownHSVImage = applyHsvOuter(srcImage, numThreads);
        });
        measureTime((char *) "Own HSV (Inner)", [&](int numThreads) {
            ownHSVImage = applyHsvInner(srcImage, numThreads);
        });
        measureTime((char *) "Own HSV (Both)", [&](int numThreads) {
            ownHSVImage = applyHsvBoth(srcImage, numThreads);
        });
        measureTime((char *) "OpenCV HSV", [&](int numThreads) {
            cvHSVImage = applyOpenCvHsv(srcImage);
        });


        /* ---------- Emboss ---------- */

        measureTime((char *) "Own Emboss (Outer)", [&](int numThreads) {
            ownEmbossImage = applyEmbossOuter(srcImage, numThreads);
        });
        measureTime((char *) "Own Emboss (Inner)", [&](int numThreads) {
            ownEmbossImage = applyEmbossInner(srcImage, numThreads);
        });
        measureTime((char *) "Own Emboss (Both)", [&](int numThreads) {
            ownEmbossImage = applyEmbossBoth(srcImage, numThreads);
        });

        cv::imwrite("resources/images/results/own-grayscale.png", ownGrayscaleImage);
        cv::imwrite("resources/images/results/cv-grayscale.png", cvGrayscaleImage);

        cv::imwrite("resources/images/results/own-hsv.png", ownHSVImage);
        cv::imwrite("resources/images/results/cv-hsv.png", cvHSVImage);

        cv::imwrite("resources/images/results/own-emboss.png", ownEmbossImage);

        // imshow("Source Image", srcImage);
        // imshow("Own Grayscale", ownGrayscaleImage);
        // imshow("CV Grayscale", cvGrayscaleImage);
        // imshow("Difference Grayscale", abs(cvGrayscaleImage - ownGrayscaleImage));

        // cv::Mat diff = abs(cvHSVImage - ownHSVImage);
        // auto error = cv::sum(diff) / (srcImage.rows * srcImage.cols);
        // std::cout << "Error: " << error << "\n";

        // imshow("Own HSV", ownHSVImage);
        // imshow("CV HSV", cvHSVImage);
        // imshow("Difference HSV", abs(cvHSVImage - ownHSVImage));


        // imshow("Own Emboss", ownEmbossImage);
    }

    cv::waitKey(0);


    return 0;

}


