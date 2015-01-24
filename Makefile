CC=gcc
LDFLAGS=-lssl -lcrypto

all: data_dedup.o deduplicate.o displayjournal.o reassemble.o
	sh prepare.sh
	$(CC) data_dedup.o deduplicate.o -o deduplicate $(LDFLAGS) -Wall
	$(CC) data_dedup.o displayjournal.o -o displayjournal $(LDFLAGS) -Wall
	$(CC) data_dedup.o reassemble.o -o reassemble $(LDFLAGS) -Wall

cuda:	data_dedup_cuda.cu deduplicate.c
	sh prepare.sh 
	cp deduplicate.c deduplicate.cu
	cp data_dedup.h data_dedup.cuh
	nvcc -DUSE_CUDA deduplicate.cu data_dedup.c data_dedup_cuda.cu -o deduplicateGPU $(LDFLAGS) 

host: data_dedup.o deduplicate.o
	$(CC) deduplicate.o data_dedup.o -o deduplicateHost

data_dedup.o: data_dedup.c
	$(CC) -c data_dedup.c $(LDFLAGS)

deduplicate.o: deduplicate.c data_dedup.o data_dedup_cuda.o
	$(CC) -c deduplicate.c $(LDFLAGS)

data_dedup_cuda.o: data_dedup_cuda.cu
	nvcc -c data_dedup_cuda.cu

displayjournal.o: displayjournal.c data_dedup.o
	$(CC) -c displayjournal.c $(LDFLAGS)

reassemble.o: reassemble.c data_dedup.o
	$(CC) -c reassemble.c $(LDFLAGS)

clean:
	rm -f *.o
	rm -f *~
	rm -f data_dedup.cuh deduplicate.cu
	
reset:
	rm -rf metafiles/* restored/* data/* 2>/dev/null
