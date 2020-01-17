#include <cstdio>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cxxopts.hpp>

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
            auto srcPixel = srcImage.at<cv::Vec4b>(row, col);

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
            auto srcPixel = srcImage.at<cv::Vec4b>(row, col);

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

cv::Mat applyGrayscaleCollapse(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    auto destImage = cv::Mat(srcImage.rows, srcImage.cols, CV_8UC1);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for collapse(2) default(none) shared(srcImage, destImage)
    for (int row = 0; row < srcImage.rows; row++) {
        for (int col = 0; col < srcImage.cols; col++) {
            auto srcPixel = srcImage.at<cv::Vec4b>(row, col);

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
            auto srcPixel = srcImage.at<cv::Vec4b>(row, col);

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
            auto srcPixel = srcImage.at<cv::Vec4b>(row, col);

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

cv::Mat applyHsvCollapse(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC3);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for collapse(2) default(none) shared(srcImage, destImage)
    for (int row = 0; row < srcImage.rows; row++) {
        for (int col = 0; col < srcImage.cols; col++) {
            auto srcPixel = srcImage.at<cv::Vec4b>(row, col);

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
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC4);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for default(none) shared(srcImage, destImage)
    for (int row = 0; row < srcImage.rows; row++) {
        for (int col = 0; col < srcImage.cols; col++) {
            int diffR, diffG, diffB, diff, gray;
            auto srcPixel = srcImage.at<cv::Vec4b>(row, col);

            if (row == 0 || col == 0) {
                diffR = srcPixel[2];
                diffG = srcPixel[1];
                diffB = srcPixel[0];
            } else {
                auto upperLeftPixel = srcImage.at<cv::Vec4b>(row - 1, col - 1);
                diffR = std::abs(srcPixel[2] - upperLeftPixel[2]);
                diffG = std::abs(srcPixel[1] - upperLeftPixel[1]);
                diffB = std::abs(srcPixel[0] - upperLeftPixel[0]);
            }

            diff = std::max(diffR, std::max(diffG, diffB));

            gray = 128 + diff;
            gray = gray > 255 ? 255 : gray;
            gray = gray < 0 ? 0 : gray;

            cv::Vec4b destPixel = cv::Vec4b(
                    uchar(gray),
                    uchar(gray),
                    uchar(gray),
                    srcPixel[3]
            );

            destImage.at<cv::Vec4b>(row, col) = destPixel;
        }
    }

    return destImage;
}

cv::Mat applyEmbossInner(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC4);
    omp_set_num_threads(numThreads);

    for (int row = 0; row < srcImage.rows; row++) {
        #pragma omp parallel for default(none) shared(srcImage, destImage, row)
        for (int col = 0; col < srcImage.cols; col++) {
            int diffR, diffG, diffB, diff, gray;
            auto srcPixel = srcImage.at<cv::Vec4b>(row, col);

            if (row == 0 || col == 0) {
                diffR = srcPixel[2];
                diffG = srcPixel[1];
                diffB = srcPixel[0];
            } else {
                auto upperLeftPixel = srcImage.at<cv::Vec4b>(row - 1, col - 1);
                diffR = std::abs(srcPixel[2] - upperLeftPixel[2]);
                diffG = std::abs(srcPixel[1] - upperLeftPixel[1]);
                diffB = std::abs(srcPixel[0] - upperLeftPixel[0]);
            }

            diff = std::max(diffR, std::max(diffG, diffB));

            gray = 128 + diff;
            gray = gray > 255 ? 255 : gray;
            gray = gray < 0 ? 0 : gray;

            cv::Vec4b destPixel = cv::Vec4b(
                    uchar(gray),
                    uchar(gray),
                    uchar(gray),
                    srcPixel[3]
            );

            destImage.at<cv::Vec4b>(row, col) = destPixel;
        }
    }

    return destImage;
}

cv::Mat applyEmbossCollapse(cv::Mat srcImage, int numThreads = omp_get_num_procs()) {
    cv::Mat destImage(srcImage.rows, srcImage.cols, CV_8UC4);
    omp_set_num_threads(numThreads);

    #pragma omp parallel for collapse(2) default(none) shared(srcImage, destImage)
    for (int row = 0; row < srcImage.rows; row++) {
        for (int col = 0; col < srcImage.cols; col++) {
            int diffR, diffG, diffB, diff, gray;
            auto srcPixel = srcImage.at<cv::Vec4b>(row, col);


            if (row == 0 || col == 0) {
                diffR = srcPixel[2];
                diffG = srcPixel[1];
                diffB = srcPixel[0];
            } else {
                auto upperLeftPixel = srcImage.at<cv::Vec4b>(row - 1, col - 1);
                diffR = std::abs(srcPixel[2] - upperLeftPixel[2]);
                diffG = std::abs(srcPixel[1] - upperLeftPixel[1]);
                diffB = std::abs(srcPixel[0] - upperLeftPixel[0]);
            }

            diff = std::max(diffR, std::max(diffG, diffB));

            gray = 128 + diff;
            gray = gray > 255 ? 255 : gray;
            gray = gray < 0 ? 0 : gray;

            cv::Vec4b destPixel = cv::Vec4b(
                    uchar(gray),
                    uchar(gray),
                    uchar(gray),
                    srcPixel[3]
            );

            destImage.at<cv::Vec4b>(row, col) = destPixel;
        }
    }

    return destImage;
}

cv::Mat applyOpenCvGrayscale(const cv::Mat &srcImage) {
    cv::Mat destImage;
    cv::cvtColor(srcImage, destImage, cv::COLOR_BGRA2GRAY);

    return destImage;
}

cv::Mat applyOpenCvHsv(const cv::Mat &srcImage) {
    cv::Mat destImage;
    cv::cvtColor(srcImage, destImage, cv::COLOR_BGR2HSV);
    return destImage;
}

template<typename F>
void measureTime(char *title, F func, int iterations = 1, int maxThreadCount = 16) {
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

int main(int argc, char *argv[]) {
    int iterations = 1;
    bool showResults = false;
    auto imagePath = std::string("resources/images/");
    auto imageName = std::string();

    cxxopts::Options options("openmp_emboss", "Image conversion with OMP: Grayscale, HSV and Emboss");
    options.add_options()
            ("f,file", "Image to convert", cxxopts::value<std::string>(imageName), "image")
            ("p,path", "Path to the image folder", cxxopts::value<std::string>(imagePath), "path")
            ("i,iterations", "Iterations for testing", cxxopts::value<int>(iterations), "number")
            ("r,results", "Show results", cxxopts::value<bool>(showResults));
    auto results = options.parse(argc, argv);

    if (results.count("file") == 1) {
        std::cout << "Running algorithm-tests with " << iterations << " iteration(s)" << std::endl;
        std::cout << "-------------------------------------------------" << std::endl << std::endl << std::endl;

        std::string imageFullPath = std::string(imagePath) + std::string(imageName);

        cv::Mat srcImage = cv::imread(imageFullPath, cv::IMREAD_UNCHANGED);
        cv::Mat myGrayscaleImage;
        cv::Mat cvGrayscaleImage;

        cv::Mat myHSVImage;
        cv::Mat cvHSVImage;

        cv::Mat myEmbossImage;

        if(srcImage.channels() == 3) {
            cv::cvtColor(srcImage, srcImage, cv::COLOR_BGR2BGRA);
        } else if(srcImage.channels() == 1) {
            cv::cvtColor(srcImage, srcImage, cv::COLOR_GRAY2BGRA);
        }
        
        if (!srcImage.data) {
            printf("No srcImage data\n");
        } else {
            /* ---------- Grayscale ---------- */

            measureTime((char *) "My Grayscale (Outer)", [&](int numThreads) {
                applyGrayscaleOuter(srcImage, numThreads);
            }, iterations);
            measureTime((char *) "My Grayscale (Inner)", [&](int numThreads) {
                applyGrayscaleInner(srcImage, numThreads);
            }, iterations);
            measureTime((char *) "My Grayscale (Collapse)", [&](int numThreads) {
                myGrayscaleImage = applyGrayscaleCollapse(srcImage, numThreads);
            }, iterations);
            measureTime((char *) "OpenCV Grayscale", [&](int numThreads) {
                cvGrayscaleImage = applyOpenCvGrayscale(srcImage);
            }, iterations);

            /* ---------- HSV ---------- */

            measureTime((char *) "My HSV (Outer)", [&](int numThreads) {
                applyHsvOuter(srcImage, numThreads);
            }, iterations);
            measureTime((char *) "My HSV (Inner)", [&](int numThreads) {
                applyHsvInner(srcImage, numThreads);
            }, iterations);
            measureTime((char *) "My HSV (Collapse)", [&](int numThreads) {
                myHSVImage = applyHsvCollapse(srcImage, numThreads);
            }, iterations);
            measureTime((char *) "OpenCV HSV", [&](int numThreads) {
                cvHSVImage = applyOpenCvHsv(srcImage);
            }, iterations);


            /* ---------- Emboss ---------- */

            measureTime((char *) "My Emboss (Outer)", [&](int numThreads) {
                applyEmbossOuter(srcImage, numThreads);
            }, iterations);
            measureTime((char *) "My Emboss (Inner)", [&](int numThreads) {
                applyEmbossInner(srcImage, numThreads);
            }, iterations);
            measureTime((char *) "My Emboss (Collapse)", [&](int numThreads) {
                myEmbossImage = applyEmbossCollapse(srcImage, numThreads);
            }, iterations);


            std::cout << "Saving my grayscale" << std::endl;
            cv::imwrite("resources/images/results/grayscale-my." + imageName, myGrayscaleImage);
            std::cout << "Saving cv grayscale" << std::endl;
            cv::imwrite("resources/images/results/grayscale-cv." + imageName, cvGrayscaleImage);

            std::cout << "Saving my hsv" << std::endl;
            cv::imwrite("resources/images/results/hsv-my." + imageName, myHSVImage);
            std::cout << "Saving cv hsv" << std::endl;
            cv::imwrite("resources/images/results/hsv-cv." + imageName, cvHSVImage);

            std::cout << "Saving my emboss" << std::endl;
            cv::imwrite("resources/images/results/emboss-my." + imageName, myEmbossImage);


            if (showResults) {
                imshow("Source Image", srcImage);
                imshow("My Grayscale", myGrayscaleImage);
                imshow("CV Grayscale", cvGrayscaleImage);
                // imshow("Difference Grayscale", abs(cvGrayscaleImage - myGrayscaleImage));

                // cv::Mat diff = abs(cvHSVImage - myHSVImage);
                // auto error = cv::sum(diff) / (srcImage.rows * srcImage.cols);
                // std::cout << "Error: " << error << "\n";

                imshow("My HSV", myHSVImage);
                imshow("CV HSV", cvHSVImage);
                // imshow("Difference HSV", abs(cvHSVImage - myHSVImage));

                imshow("My Emboss", myEmbossImage);
            }
        }

        cv::waitKey(0);


        return 0;
    } else {
        std::cout << options.help() << std::endl;
    }
}


