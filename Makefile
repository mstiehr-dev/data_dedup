CC=gcc
LDFLAGS=-lssl -lcrypto

all: data_dedup.o deduplicate.o displayjournal.o reassemble.o
	sh prepare.sh
	$(CC) data_dedup.o deduplicate.o -o bin/deduplicate $(LDFLAGS) -Wall
	$(CC) data_dedup.o displayjournal.o -o bin/displayjournal $(LDFLAGS) -Wall
	$(CC) data_dedup.o reassemble.o -o bin/reassemble $(LDFLAGS) -Wall


data_dedup.o: data_dedup.c data_dedup.h
	$(CC) -c data_dedup.c $(LDFLAGS)

deduplicate.o: deduplicate.c data_dedup.o
	$(CC) -c deduplicate.c $(LDFLAGS)

displayjournal.o: displayjournal.c data_dedup.o
	$(CC) -c displayjournal.c $(LDFLAGS)

reassemble.o: reassemble.c data_dedup.o
	$(CC) -c reassemble.c $(LDFLAGS)

clean:
	rm -f *.o
	rm -f *~
	
empty:
	rm -rf metafiles/* restored/* data/* 2>/dev/null
