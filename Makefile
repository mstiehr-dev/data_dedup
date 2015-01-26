CC=gcc
LDFLAGS=-lssl -lcrypto

cuda: deduplicate.c data_dedup_cuda.cu data_dedup_USECUDA.o
	sh prepare.sh 
	cp deduplicate.c deduplicate.cu
	cp data_dedup.h data_dedup.cuh
	nvcc $(LDFLAGS) -DUSE_CUDA deduplicate.cu data_dedup_USECUDA.o data_dedup_cuda.cu -o deduplicateGPU  

host: deduplicate.c data_dedup.o
	sh prepare.sh
	$(CC) $(LDFLAGS) deduplicate.c data_dedup.o -o deduplicateHost


displayjournal: 
	$(CC) $(LDFLAGS) displayjournal.c data_dedup.o -o displayjournal

reassemble:
	$(CC) $(LDFLAGS) reassemble.c data_dedup.o -o reassemble

displaymetafile: 
	$(CC) $(LDFLAGS) displaymetafile.c -o displaymetafile

data_dedup.o: data_dedup.c data_dedup.h
	$(CC) $(LDFLAGS) -c data_dedup.c

data_dedup_USECUDA.o: data_dedup.c
	cp data_dedup.c data_dedup.cu
	nvcc -lssl -lcrypto -c -DUSE_CUDA data_dedup.cu -o data_dedup_USECUDA.o

clean:
	rm -f *.o
	rm -f *~
	rm -f data_dedup.cuh deduplicate.cu
	
reset:
	rm -rf metafiles/* restored/* data/* 2>/dev/null
