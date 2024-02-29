#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "util.h"

#define MAX_PATH_LENGTH 256
#define IMAGE_WIDTH  800
#define IMAGE_HEIGHT 600

void load_image(int *image, int width, int height, const char *filename) {
}

void save_image(int *image, int width, int height, const char *filename) {
}

void process_image(int *image, int width, int height, int operation) {
}

int main(int argc, char **argv) {
    int rank, size;
    char image_path[MAX_PATH_LENGTH];
    int operation;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Uso: %s <caminho_da_imagem.bmp> <operacao>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Obter o caminho da imagem e a operação a ser realizada
    strncpy(image_path, argv[1], MAX_PATH_LENGTH);
    operation = atoi(argv[2]);

    // Carregar a imagem
    int *image = (int *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(int));
    if (rank == 0) {
        load_image(image, IMAGE_WIDTH, IMAGE_HEIGHT, image_path);
    }

    process_image(image, IMAGE_WIDTH / size, IMAGE_HEIGHT, operation);

    // Coletar as partes processadas da imagem
    // Implemente esta parte usando MPI_Gather

    // Salvar a imagem resultante
    if (rank == 0) {
        save_image(image, IMAGE_WIDTH, IMAGE_HEIGHT, "output_image.bmp");
    }

    // Liberar memória
    free(image);

    MPI_Finalize();

    return 0;
}
