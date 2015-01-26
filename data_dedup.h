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

	time_t startZeit;
	double laufZeit;

	

	#define spareEntries 10000
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

	#ifdef USE_CUDA
		static void cudaCheckError(cudaError_t error, const char *file, int line) {
			if(error!=cudaSuccess) {
				printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
				exit(EXIT_FAILURE);
			}
		}
		#define CUDA_HANDLE_ERR(err) (cudaCheckError(err,__FILE__, __LINE__))
		cudaDeviceProp prop; // zur Ermittlung der GPU Eckdaten 
		size_t totalGlobalMem; 
		size_t sharedMemPerBlock;
		int max_threadsPerBlock;
		__constant__ char goldenHash[33];	// im Constant-Cache gehaltener Such-String
		int blocks = 4;	// Konfiguration des Kernelaufrufs: Anzahl der Blöcke || beste Performance: 2* MultiProcessorCount
		int threadsPerBlock = 1024; // maximum
		
		void cudaCopyJournal(void *, void *, off_t);
		void cudaExtendHashStack(void *, journalentry *);
		#ifndef CUDA_HANDLE_ERR
			#define CUDA_HANDLE_ERR(err) (cudaCheckError(err, __FILE__, __LINE__))
		#endif
	#endif


#endif // __PSEM__

