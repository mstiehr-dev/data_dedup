CC=gcc
LDFLAGS=-lssl -lcrypto

all: data_dedup.o deduplicate.o assemble.o display_journal.o readjournal.o
	$(CC) data_dedup.o deduplicate.o -o deduplicate $(LDFLAGS)
	$(CC) data_dedup.o assemble.o -o assemble $(LDFLAGS)
	$(CC) data_dedup.o display_journal.o -o displayjournal $(LDFLAGS)
	$(CC) data_dedup.o readjournal.o -o readjournal $(LDFLAGS)


data_dedup.o: data_dedup.c data_dedup.h
	$(CC) -c data_dedup.c $(LDFLAGS)

deduplicate.o: deduplicate.c data_dedup.h
	$(CC) -c deduplicate.c $(LDFLAGS)
	rm -rf journal.* storage.* metafiles/* restored/* 2>/dev/null

assemble.o: assemble.c data_dedup.h
	$(CC) -c assemble.c $(LDFLAGS)
	
display_journal.o: display_journal.c data_dedup.h
	$(CC) -c display_journal.c $(LDFLAGS)

readjournal.o: readjournal.c data_dedup.h
	$(CC) -c readjournal.c $(LDFLAGS)

clean:
	rm *.o
	
empty:
	rm -rf metafiles/* restored/* *.dat 2>/dev/null
