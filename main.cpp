#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include "util.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            cerr << "Uso: " << argv[0] << " <caminho_para_imagem.bmp>" << endl;
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        Mat image;
        image = imread(argv[1], IMREAD_GRAYSCALE);

        if (!image.data) {
            cerr << "Erro ao abrir a imagem." << endl;
            MPI_Finalize();
            return 1;
        }

        // Filtragem com Kernel

        Mat filteredImage = image.clone();
        Mat kernel = Mat::ones(5, 5, CV_32F);
        medianFilter(filteredImage, kernel);

        // Filtragem de mediana

        Mat medianImage = image.clone();
        
        // Cisalhamento
        
        Mat shearingImage = image.clone();
        Mat cis_matrix = (Mat_<float>(2, 3) << 1, 0.5, 0, 0, 1, 0);
        transform(shearingImage, cis_matrix);
        
        imwrite("convolucao.bmp", filteredImage);
        imwrite("cisalhamento2.bmp", shearingImage);

        double executionTime = measureExecutionTime(medianFilter, image, kernel);
        cout << executionTime << endl;

        if (!image.data) {
            cerr << "Erro ao abrir a imagem." << endl;
            MPI_Finalize();
            return 1;
        }

        int cols = image.cols;
        int rows = image.rows;
        int channels = image.channels();

        cout << "Informações da imagem:" << endl;
        cout << "Largura: " << image.cols << " pixels" << endl;
        cout << "Altura: " << image.rows << " pixels" << endl;
        cout << "Número de canais: " << image.channels() << endl;

        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
    } else {
        int cols, rows, channels;
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);

        cout << "Processo " << rank << " recebeu informações da imagem:" << endl;
        cout << "Largura: " << cols << " pixels" << endl;
        cout << "Altura: " << rows << " pixels" << endl;
        cout << "Número de canais: " << channels << endl;
    }

    MPI_Finalize();
    
    return 0;
}
