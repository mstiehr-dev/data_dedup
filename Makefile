CC=gcc
LDFLAGS=-lssl -lcrypto

all: data_dedup.o deduplicate.o displayjournal.o reassemble.o
	sh prepare.sh
	$(CC) data_dedup.o deduplicate.o -o deduplicate $(LDFLAGS) -Wall
	$(CC) data_dedup.o displayjournal.o -o displayjournal $(LDFLAGS) -Wall
	$(CC) data_dedup.o reassemble.o -o reassemble $(LDFLAGS) -Wall

cuda:	data_dedup_cuda.cu deduplicate.c data_dedup.o data_dedup.h data_dedup.cuh 
	sh prepare.sh 
	cp deduplicate.c deduplicate.cu
	nvcc deduplicate.cu data_dedup.o data_dedup_cuda.cu -o deduplicateGPU $(LDFLAGS) -DUSE_CUDA

host: data_dedup.o deduplicate.o
	$(CC) deduplicate.o data_dedup.o -o deduplicateHost

data_dedup.o: data_dedup.c data_dedup.h
	$(CC) -c data_dedup.c $(LDFLAGS)

deduplicate.o: deduplicate.c data_dedup.o data_dedup_cuda.o
	$(CC) -c deduplicate.c $(LDFLAGS)

data_dedup_cuda.o: data_dedup_cuda.cu data_dedup_cuda.cuh
	nvcc data_dedup_cuda.cu -c

displayjournal.o: displayjournal.c data_dedup.o
	$(CC) -c displayjournal.c $(LDFLAGS)

reassemble.o: reassemble.c data_dedup.o
	$(CC) -c reassemble.c $(LDFLAGS)

clean:
	rm -f *.o
	rm -f *~
	
reset:
	rm -rf metafiles/* restored/* data/* 2>/dev/null
