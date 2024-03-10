#ifndef UTIL_H
#define UTIL_H
#include <iostream>
#include <functional>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;
using namespace cv;

template<typename Func, typename... Args>
double measureExecutionTime(Func func, Args&&... args) {
    
    auto start = high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = high_resolution_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(end - start);
    return time_span.count();
}

uchar mean(Mat region, int kernelSize) {
    int sum = 0;
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            sum += region.at<uchar>(i, j);
        }
    }
    return (uchar) sum / (kernelSize * kernelSize);
}

uchar median(Mat region, int kernelSize) {
    int size = kernelSize * kernelSize;
    int *values = new int[size];
    int k = 0;
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            values[k++] = region.at<uchar>(i, j);
        }
    }
    sort(values, values + size);
    return (uchar) values[size / 2];
}


void border(Mat image, int verticalOffset, int horizontalOffset, Mat result) {
    // cria borda
    #pragma omp parallel for
    for (int i = verticalOffset; i < image.rows + verticalOffset; i++) {
        for (int j = horizontalOffset; j < image.cols + horizontalOffset; j++) {
            result.at<uchar>(i, j) = image.at<uchar>(i - verticalOffset, j - horizontalOffset);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < verticalOffset; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.at<uchar>(i, j) = result.at<uchar>(verticalOffset, j);
        }
    }

    #pragma omp parallel for
    for (int i = result.rows - verticalOffset; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.at<uchar>(i, j) = result.at<uchar>(result.rows - verticalOffset - 1, j);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < horizontalOffset; j++) {
            result.at<uchar>(i, j) = result.at<uchar>(i, horizontalOffset);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < result.rows; i++) {
        for (int j = result.cols - horizontalOffset; j < result.cols; j++) {
            result.at<uchar>(i, j) = result.at<uchar>(i, result.cols - horizontalOffset - 1);
        }
    }
}

void meanFilter(Mat image, Mat kernel) {
    int kernelHeight = kernel.rows;
    int kernelWidth = kernel.cols;

    int verticalOffset = floor(kernelHeight / 2);
    int horizontalOffset = floor(kernelWidth / 2);

    Size resultSize(image.cols + horizontalOffset*2, image.rows + verticalOffset*2);

    Mat result = Mat::ones(resultSize, image.type());
    
    border(image, verticalOffset, horizontalOffset, result);

    #pragma omp parallel for
    for (int i = verticalOffset; i < image.rows + verticalOffset; i++) {
        for (int j = horizontalOffset; j < image.cols + horizontalOffset; j++) {
            Mat region = result(Rect(j - horizontalOffset, i - verticalOffset, kernelWidth, kernelHeight));
            image.at<uchar>(i - verticalOffset, j - horizontalOffset) = mean(region, kernelWidth);
        }
    }
    
}

void medianFilter(Mat image, Mat kernel) {
    int kernelHeight = kernel.rows;
    int kernelWidth = kernel.cols;

    int verticalOffset = floor(kernelHeight / 2);
    int horizontalOffset = floor(kernelWidth / 2);

    Size resultSize(image.cols + horizontalOffset*2, image.rows + verticalOffset*2);

    Mat result = Mat::ones(resultSize, image.type());
    
    border(image, verticalOffset, horizontalOffset, result);

    #pragma omp parallel for
    for (int i = verticalOffset; i < image.rows + verticalOffset; i++) {
        for (int j = horizontalOffset; j < image.cols + horizontalOffset; j++) {
            Mat region = result(Rect(j - horizontalOffset, i - verticalOffset, kernelWidth, kernelHeight));
            image.at<uchar>(i - verticalOffset, j - horizontalOffset) = median(region, kernelWidth);
        }
    }
    
}


void transform(Mat image, Mat matrix) {
    Mat transformedImage(image.rows, image.cols, image.type());

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            float x_transformed = matrix.at<float>(0, 0) * x + matrix.at<float>(0, 1) * y + matrix.at<float>(0, 2);
            float y_transformed = matrix.at<float>(1, 0) * x + matrix.at<float>(1, 1) * y + matrix.at<float>(1, 2);

            if (x_transformed >= 0 && x_transformed < image.cols && y_transformed >= 0 && y_transformed < image.rows) {
                int x1 = floor(x_transformed);
                int y1 = floor(y_transformed);
                int x2 = x1 + 1;
                int y2 = y1 + 1;

                float alpha = x_transformed - x1;
                float beta = y_transformed - y1;

                float interpolated_value = 

                (1 - alpha) * (1 - beta) * image.at<uchar>(y1, x1) + 
                alpha * (1 - beta) * image.at<uchar>(y1, x2) + 
                (1 - alpha) * beta * image.at<uchar>(y2, x1) +
                alpha * beta * image.at<uchar>(y2, x2);

                transformedImage.at<uchar>(y, x) = (uchar)interpolated_value;
            }
        }
    }

    image = transformedImage;
}

#endif