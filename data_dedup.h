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


	#define CHUNKSIZE 512

	// allgemeine dateien 
	#define JOURNALFILE "./journal.dat"
	#define STORAGEDUMP "./storage.dat"
	#define RESTOREDIR "./restored/"
	#define METADIR "./metafiles/"

	#define TRUE  1
	#define FALSE 0

	struct datensatz {
		long blocknummer;
		char hash[33];
		short length;
	};

	#define JOURNALLINELENGTH sizeof(struct datensatz) //sizeof(long) + 33*sizeof(char) + sizeof(unsigned short)
#define EXTRASPACEFORNEWTUPELS 100*JOURNALLINELENGTH
	
	struct stat inputfile_statbuffer;

	size_t addedTupels;

	// Funktionen: 
	long   isHashInJournal(char *, FILE *);
	long   isHashInMappedJournal(char *, void *, long );
	char * buildString3s(const char *, const char *, const char *);

	// Variablen
	long inputFileSize;
	long journalLineNumber; // Zeilenindex des Journals
	long hashInJournal; // nummer der zeile, in der ein hash steht
	long hashIDforMetafile;
	long storageBlockPosition; // Anfang des Datenblocks im Storagedump
	long blockLength;
	char *    journalLineBuffer;
	char *    filename;
	char *    metafilename;
	char *	  restorefilename;
	size_t journalLineLength;// = sizeof(size_t) + 33*sizeof(char) + sizeof(long);
	int 	  retVal;

	

#endif
