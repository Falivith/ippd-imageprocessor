#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <functional>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace std::filesystem;

#define MAX_THREADS 4

Mat createHighlightKernel(int size) {
    CV_Assert(size % 2 == 1); // Size should be odd to have a central element
    Mat kernel = Mat::ones(size, size, CV_32F) / (float)(size * size);
    float centerValue = 0.5f; // You can adjust the center value as needed
    kernel.at<float>(size / 2, size / 2) = centerValue;
    return kernel;
}

void write_csv(const string& filename, const string& image_name, const string& process, int kernel_size, double executionTime) {
    ofstream file;
    bool file_exists = exists(filename);
    file.open(filename, ios::app);

    if (!file_exists) {
        file << "image,type,kernel_size,time\n";
    }

    file << image_name << "," << process << "," << kernel_size << "," << executionTime << "\n";

    file.close();
}

void load_image_names(vector<string> &image_names) {
    try {
        for (const auto &entry : directory_iterator("images")) {
            if (entry.is_regular_file()) {
                image_names.push_back(entry.path().string());
            }
        }
    } catch (const filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "General error: " << e.what() << '\n';
    }
}

template<typename Func, typename... Args>
double measureExecutionTime(Func func, Args&&... args) {
    
    auto start = high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(end - start);
    return time_span.count();
}

float mean(Mat region, int kernelSize) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum) collapse(2)
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            sum += region.at<uchar>(i, j);
            cout << sum << endl;
        }
    }
    return sum / (kernelSize * kernelSize);
}

float median(Mat region, int kernelSize) {
    int size = kernelSize * kernelSize;
    int *values = new int[size];
    int k = 0;
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            values[k++] = region.at<uchar>(i, j);
        }
    }
    sort(values, values + size);
    return values[size / 2];
}


void border(const Mat& image, int offset, Mat& result) {
    int newRows = image.rows + 2 * offset;
    int newCols = image.cols + 2 * offset;
    result = Mat::zeros(newRows, newCols, image.type());

    #pragma omp parallel for collapse(2) num_threads(MAX_THREADS)
    for (int i = offset; i < image.rows + offset; i++) {
        for (int j = offset; j < image.cols + offset; j++) {
            result.at<uchar>(i, j) = image.at<uchar>(i - offset, j - offset);
        }
    }

    #pragma omp parallel for collapse(2) num_threads(MAX_THREADS)
    for (int i = 0; i < offset; i++) {
        for (int j = 0; j < newCols; j++) {
            result.at<uchar>(i, j) = result.at<uchar>(offset, j);
            result.at<uchar>(newRows - 1 - i, j) = result.at<uchar>(newRows - 1 - offset, j);
        }
    }

    #pragma omp parallel for collapse(2) num_threads(MAX_THREADS)
    for (int i = 0; i < newRows; i++) {
        for (int j = 0; j < offset; j++) {
            result.at<uchar>(i, j) = result.at<uchar>(i, offset);
            result.at<uchar>(i, newCols - 1 - j) = result.at<uchar>(i, newCols - 1 - offset);
        }
    }
}

void convolute(Mat& image, const Mat& kernel) {
    CV_Assert(image.channels() == 1);

    int offset = kernel.rows / 2;
    Mat paddedImage;
    border(image, offset, paddedImage);

    Mat result = Mat::zeros(image.size(), image.type());

    #pragma omp parallel for collapse(2)
    for (int i = offset; i < paddedImage.rows - offset; ++i) {
        for (int j = offset; j < paddedImage.cols - offset; ++j) {
            float sum = 0.0f;
            for (int k = -offset; k <= offset; ++k) {
                for (int l = -offset; l <= offset; ++l) {
                    sum += paddedImage.at<uchar>(i + k, j + l) * kernel.at<float>(k + offset, l + offset);
                }
            }
            sum = std::min(std::max(sum, 0.0f), 255.0f);
            result.at<uchar>(i - offset, j - offset) = static_cast<uchar>(sum);
        }
    }
    image = result;
}

void meanFilter(Mat& image, int kernel_size) {
    Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size * kernel_size);
    convolute(image, kernel);
}

void medianFilter(Mat image, int kernel_size) {
    int offset = floor(kernel_size / 2);

    Size resultSize(image.cols + offset*2, image.rows + offset*2);

    Mat result = Mat::ones(resultSize, image.type());
    
    border(image, offset, result);

    #pragma omp parallel for collapse(2) num_threads(MAX_THREADS)
    for (int i = offset; i < image.rows + offset; i++) {
        for (int j = offset; j < image.cols + offset; j++) {
            Mat region = result(Rect(j - offset, i - offset, kernel_size, kernel_size));
            image.at<uchar>(i - offset, j - offset) = median(region, kernel_size);
        }
    }
}

void transformImage(Mat& image, const Mat& matrix) {
    Size newSize = image.size();
    Mat transformedImage = Mat::zeros(newSize, image.type());

    Point2f srcCenter(image.cols / 2.0F, image.rows / 2.0F);
    Point2f dstCenter(newSize.width / 2.0F, newSize.height / 2.0F);

    #pragma omp parallel for collapse(2) num_threads(MAX_THREADS)
    for (int y = 0; y < newSize.height; y++) {
        for (int x = 0; x < newSize.width; x++) {
            Mat dstCoords = (Mat_<float>(3, 1) << x - dstCenter.x, y - dstCenter.y, 1);
            Mat srcCoords = matrix * dstCoords;

            float x_src = srcCoords.at<float>(0, 0) + srcCenter.x;
            float y_src = srcCoords.at<float>(1, 0) + srcCenter.y;

            if (x_src >= 0 && x_src < image.cols && y_src >= 0 && y_src < image.rows) {
                transformedImage.at<uchar>(y, x) = image.at<uchar>((int)y_src, (int)x_src);
            }
        }
    }

    image = transformedImage;
}

#endif