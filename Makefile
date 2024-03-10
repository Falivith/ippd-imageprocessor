all:
	mpic++ -o main main.cpp `pkg-config --cflags --libs opencv4` -I/usr/include/mpi -lmpi -I. -fopenmp
	mpirun -np 6 ./main images/image.png
	rm main

compile:
	mpic++ -o main main.cpp `pkg-config --cflags --libs opencv4` -I/usr/include/mpi -lmpi -I.

run:
	mpirun -np 6 ./main images/image.png

clean:
	rm main

