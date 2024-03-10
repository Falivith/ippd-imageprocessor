all:
	mpic++ -o main main.cpp `pkg-config --cflags --libs opencv4` -I/usr/include/mpi -lmpi -I.

run:
	mpirun -np 2 ./main images/saltpepper1.png

clean:
	rm main

