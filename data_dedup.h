/* data_dedup.h */
#ifndef __PSEM__

/* Makros: 
 * USE_CUDA
 */

	#define __PSEM__
	#ifndef _GNU_SOURCE
		#define _GNU_SOURCE
	#endif
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
	#include <time.h>
	#include <stddef.h>
	#include <limits.h>

	#define CHUNKSIZE 512

	// allgemeine dateien 
	#define JOURNALFILE "./data/journal.dat"
	#define STORAGEDUMP "./data/storage.dat"
	#define RESTOREDIR "./restored/"
	#define METADIR "./metafiles/"

	#define TRUE  1
	#define FALSE 0

	typedef struct {
		long  block;
		char  hash[32+1];
		short len;
	} journalentry;

	

	

	#define spareEntries 100000 // 100_000
	#define auxSpace spareEntries*sizeof(journalentry) // wird verwendet bei MapFile(), gibt an wieviel zusätzlicher Platz reserviert wird 

	// FUNKTIONEN 
	const char * buildString3s(const char *, const char *, const char *);
	void  * mapFile(int fd, off_t len, int aux, off_t *saveLen);
	long 	isHashInMappedJournal(char *hash, void * add, long records);
	long 	isHashInJournalGPU(char *, void *, off_t);
	char   	getRandChar();
	char  *	getRandString(size_t n);
	float  	getRandFloat();
	int 	memcmp4l(char *, char *);



#endif // __PSEM__

