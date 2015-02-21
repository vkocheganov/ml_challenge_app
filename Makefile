CC=g++ -O3
CFLAGS=-c -std=c++11
LIBS=
all: opencv_test

opencv_test: main.o io.o models.o properties.o
	$(CC) main.o io.o models.o properties.o  -o opencv_test -lopencv_core -lopencv_ml
#opencv_test: main.o io.o models.o properties.o
#	$(CC) main.o io.o models.o properties.o  -o opencv_test -L. -llib1 -llib2

main.o: main.cpp 
	$(CC) $(CFLAGS) main.cpp

io.o: io.cpp
	$(CC) $(CFLAGS) io.cpp

models.o: models.cpp
	$(CC) $(CFLAGS) models.cpp

properties.o: properties.cpp
	$(CC) $(CFLAGS) properties.cpp

clean:
	rm -rf *.o opencv_test
