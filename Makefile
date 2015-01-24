CC=gcc
LDFLAGS=-lssl -lcrypto

cuda:	
	sh prepare.sh 
	cp deduplicate.c deduplicate.cu
	cp data_dedup.h data_dedup.cuh
	nvcc $(LDFLAGS) -DUSE_CUDA deduplicate.cu data_dedup.c data_dedup_cuda.cu -o deduplicateGPU  

host: 
	sh prepare.sh
	$(CC) $(LDFLAGS) deduplicate.c data_dedup.c -o deduplicateHost


displayjournal: 
	$(CC) $(LDFLAGS) displayjournal.c data_dedup.c -o displayjournal

reassemble:
	$(CC) $(LDFLAGS) reassemble.c data_dedup.c -o reassemble

clean:
	rm -f *.o
	rm -f *~
	rm -f data_dedup.cuh deduplicate.cu
	
reset:
	rm -rf metafiles/* restored/* data/* 2>/dev/null
