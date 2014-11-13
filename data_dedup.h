/* data_dedup.h */
#ifndef __ProjSem_


	#define __ProjSem_
	#define _GNU_SOURCE
	#include <stdio.h>
	#include <stdlib.h>
	#include <unistd.h>
	#include <sys/stat.h>
	#include <sys/types.h>
	#include <string.h>
	#include <libgen.h> /* basename() */
	#include <fcntl.h>
	#include <openssl/md5.h>
	#include <sys/mman.h>
	#include <errno.h>


	#define CHUNKSIZE 512

	// allgemeine dateien 
	#define JOURNALFILE "./data/journal.dat"
	#define STORAGEDUMP "./data/storage.dat"
	#define RESTOREDIR "./restored/"
	#define METADIR "./metafiles/"

	#define TRUE  1
	#define FALSE 0

	typedef struct {
		long block;
		char hash[32+1];
		short len;
	} journalentry;

	#define spareEntries 10000
	#define auxSpace spareEntries*sizeof(journalentry)

	// FUNKTIONEN 
	void * mapFile(int fd, off_t len, int aux, off_t *saveLen);

#endif

/*
	// Funktionen: 
	long   isHashInMappedJournal(char *, void *, long );
	char * buildString3s(const char *, const char *, const char *);

	// Variablen
	long inputFileSize;
	long journalLineNumber; // Zeilenindex des Journals
	long hashIDforMetafile;
	long storageBlockPosition; // Anfang des Datenblocks im Storagedump
	long blockLength;
	char *    journalLineBuffer;
	char *    filename;
	char *    metafilename;
	char *	  restorefilename;
	size_t journalLineLength;// = sizeof(size_t) + 33*sizeof(char) + sizeof(long);
	int 	  retVal;

	*/

