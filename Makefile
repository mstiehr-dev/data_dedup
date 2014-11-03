CC=gcc
LDFLAGS=-lssl -lcrypto

all: data_dedup.o deduplicate.o assemble.o
	$(CC) data_dedup.o deduplicate.o -o deduplicate $(LDFLAGS)
	$(CC) data_dedup.o assemble.o -o assemble $(LDFLAGS)

data_dedup.o: data_dedup.c data_dedup.h
	$(CC) -c data_dedup.c $(LDFLAGS)

deduplicate.o: deduplicate.c data_dedup.h
	$(CC) -c deduplicate.c $(LDFLAGS)
	#rm -rf journal.txt storage.dump 2>/dev/null

assemble.o: assemble.c data_dedup.h
	$(CC) -c assemble.c $(LDFLAGS)

clean:
	rm *.o
