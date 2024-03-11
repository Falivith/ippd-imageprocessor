all:
	mpic++ -o main main.cpp `pkg-config --cflags --libs opencv4` -I/usr/include/mpi -lmpi -I. -fopenmp
	mpirun -np 6 ./main
	rm main

compile:
	mpic++ -o main main.cpp `pkg-config --cflags --libs opencv4` -I/usr/include/mpi -lmpi -I.

run:
	mpirun -np 3 ./main

clean:
	rm main

