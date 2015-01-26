CC=gcc
LDFLAGS=-lssl -lcrypto

cuda: deduplicate.c
	sh prepare.sh 
	cp deduplicate.c deduplicate.cu
	nvcc -lssl -lcrypto -DUSE_CUDA deduplicate.cu -o deduplicateGPU  

host: deduplicate.c
	sh prepare.sh
	$(CC) $(LDFLAGS) deduplicate.c -o deduplicateHost


displayjournal: 
	$(CC) $(LDFLAGS) displayjournal.c data_dedup.o -o displayjournal

reassemble:
	$(CC) $(LDFLAGS) reassemble.c data_dedup.o -o reassemble

displaymetafile: 
	$(CC) $(LDFLAGS) displaymetafile.c -o displaymetafile

clean:
	rm -f *.o
	rm -f *~
	rm -f data_dedup.cuh deduplicate.cu
	
reset:
	rm -rf metafiles/* restored/* data/* 2>/dev/null
