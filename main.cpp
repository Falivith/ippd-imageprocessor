#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include "util.h"
#include <filesystem>

using namespace std;
using namespace cv;
using namespace std::filesystem;

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::string> image_names;
    load_image_names(image_names);

    // Distribuir o número total de imagens entre os processos
    int images_per_process = image_names.size() / size;
    int remainder = image_names.size() % size;

    std::vector<std::string> local_image_names;
    if (rank < remainder) {
        local_image_names.assign(image_names.begin() + rank * (images_per_process + 1),
                                  image_names.begin() + (rank + 1) * (images_per_process + 1));
    } else {
        local_image_names.assign(image_names.begin() + rank * images_per_process + remainder,
                                  image_names.begin() + (rank + 1) * images_per_process + remainder);
    }

    // Carregar as imagens locais
    vector<Mat> images;
    for (string image_name : local_image_names) {
        images.push_back(imread(image_name, IMREAD_GRAYSCALE));
    }

    vector<int> kernel_sizes = {3, 5, 7, 9, 11};
    string result_folder = "result_images/";
    string csv_filename = "image_processing_results.csv";
    double executionTime;

    for (int kernel_size : kernel_sizes) {
        for (size_t i = 0; i < images.size(); ++i) {
            Mat& image = images[i];
            string& image_name = local_image_names[i];
            size_t lastSlashPos = image_name.find_last_of('/');
            size_t dotPos = image_name.find_last_of('.');
            string number = image_name.substr(lastSlashPos + 1, dotPos - lastSlashPos - 1);

            cout << "Process " << rank << " processing image " << image_name << " with kernel size " << kernel_size << endl;

            Mat medianImage = image.clone();
            executionTime = measureExecutionTime(medianFilter, medianImage, kernel_size);
            write_csv(csv_filename, number, "median", kernel_size, executionTime);
            imwrite(result_folder + number + "_median_kernel_size_" + to_string(kernel_size) + ".bmp", medianImage);

            Mat meanImage = image.clone();
            executionTime = measureExecutionTime(meanFilter, meanImage, kernel_size);
            write_csv(csv_filename, number, "mean", kernel_size, executionTime);
            imwrite(result_folder + number + "_mean_kernel_size_" + to_string(kernel_size) + ".bmp", meanImage);

            Mat highlightKernel = createHighlightKernel(kernel_size);
            Mat highlightedImage = image.clone();
            executionTime = measureExecutionTime(convolute, highlightedImage, highlightKernel);
            write_csv(csv_filename, number, "highlight", kernel_size, executionTime);
            imwrite(result_folder + number + "_highlighted_kernel_size" + to_string(kernel_size) + ".bmp", highlightedImage);
        }
    }

    for (int i = 0; i < images.size(); i++) {
        Mat image = images[i];
        string image_name = local_image_names[i];
        size_t lastSlashPos = image_name.find_last_of('/');
        size_t dotPos = image_name.find_last_of('.');
        string number = image_name.substr(lastSlashPos + 1, dotPos - lastSlashPos - 1);
        
        cout << "Process " << rank << " processing image " << image_name << " with rotation transformation " << endl;

        Mat rotatedImage = image.clone();
        Mat rotationMatrix = (Mat_<float>(3, 3) << 0, -1, 0, -1, 0, 0, 0, 0, 1);
        executionTime = measureExecutionTime(transformImage, rotatedImage, rotationMatrix);
        write_csv(csv_filename, number, "rotate", 0, executionTime);
        imwrite(result_folder + number + "_rotated.bmp", rotatedImage);

        cout << "Process " << rank << " processing image " << image_name << " with schear transformation " << endl;

        Mat schearImage = image.clone();
        Mat schearMatrix = (Mat_<float>(3, 3) << 1, 0.1, 0, 0.3, 1, 0, 0, 0, 1);
        executionTime = measureExecutionTime(transformImage, schearImage, schearMatrix);
        write_csv(csv_filename, number, "schear", 0, executionTime);
        imwrite(result_folder + number + "_schear.bmp", schearImage);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    MPI_Finalize();

    double total_time = end_time - start_time;

    if (rank == 0) {
        write_csv("batch_processing_results.csv", "image_processing", "a", 0, total_time);
        cout << "Total execution time: " << total_time << " seconds" << endl;
    }
    
    return 0;
}
