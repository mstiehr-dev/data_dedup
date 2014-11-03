/* data_dedup.h */
#ifndef __ProjSem_


	#define __ProjSem_

	#include <stdio.h>
	#include <stdlib.h>
	#include <unistd.h>
	#include <sys/stat.h>
	#include <sys/types.h>
	#include <string.h>
	#include <libgen.h> /* basename() */

	#include <openssl/md5.h>


	#define CHUNKSIZE 512

	// allgemeine dateien 
	#define JOURNALFILE "./journal.txt"
	#define STORAGEDUMP "./storage.dump"
	#define RESTOREDIR "./restored/"
	#define METADIR "./metafiles/"

	#define TRUE  1
	#define FALSE 0

	struct stat inputfile_statbuffer;


	/* Anzahl der existenten Blöcke ermitteln */
	long countLines(FILE *);
	long findHashInJournal(char *, FILE *);
	char * 	  buildString3s(const char *, const char *, const char *);

	// long storageBlockNummer; // laufende nummer der (globalen) blöcke // siehe storageBlockPosition
	long inputFileSize;
	long journalLineNumber; // Zeilenindex des Journals
	long hashNummer; // nummer der zeile, in der ein hash steht
	long schreibNummer;
	long storageBlockPosition; // Anfang des Datenblocks im Storagedump
	long blockLength;
	char *    dataBuffer;
	char *    filename;
	char *    metafile;
	char *	  restorefile;

	int 	  retVal;


#endif
