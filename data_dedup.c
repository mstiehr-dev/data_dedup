/* data_dedup.c */

#ifdef USE_CUDA
	#include "data_dedup.cuh"
	__constant__ char goldenHash[33];	// im Constant-Cache gehaltener Such-String
	int blocks = 4;	// Konfiguration des Kernelaufrufs: Anzahl der Blöcke || beste Performance: 2* MultiProcessorCount
	int threadsPerBlock = 1024; // maximum
#else
	#include "data_dedup.h"
#endif // USE_CUDA

const char * buildString3s(const char *s1, const char *s2, const char *s3) {
	size_t l1 = strlen(s1);
	size_t l2 = strlen(s2);
	size_t l3 = strlen(s3);
	char *newStr = (char *) malloc(sizeof(char)*(l1+l2+l3+1));
	strncpy(newStr, s1, l1);
	strncpy(newStr+l1,s2,l2);
	strncpy(newStr+l1+l2,s3,l3);
	return newStr;
}

long isHashInMappedJournal(char *hash, void * add, long records) {
	/* Rückgabewert: Zeilennummer, in der der Hash gefunden wurde, also auch die Blocknummer im dumpfile
	 * sonst -1 */
	journalentry *tupel = (journalentry *) add; // zeigt nun auf den ersten Datensatz
	unsigned long line = 0;
	while(line<records) {
		/*
		memcpy(&tupel,tempAdd,sizeof(journalentry)); 
		if(strstr(tupel.hash,hash)!=NULL) {
			// Hash gefunden
			return line;
		}
		line++;
		tempAdd += sizeof(journalentry); */
		// besser: 
		if(memcmp4l(tupel->hash, hash)==0) {
			// TREFFER!
			return line;
		} /* else */
		line++;
		tupel++;
	}
	return -1;
}

int memcmp4l(char *s, char *t) { // gibt 1 zurück bei Unterscheidung
	int i = 32/sizeof(long); // 4
	long *l1 = (long*)s;
	long *l2 = (long*)t;
	while(i--) {
		if(*l1!=*l2)
			return 1;
		l1++;
		l2++;
	}
	return 0;
}

void * mapFile(int fd, off_t len, int aux, off_t *saveLen) {
	off_t tempLen = len+aux;
	if(ftruncate(fd,tempLen)==-1) {
		perror("ftruncate()");
		exit(1);
	}
	void *add = mmap(NULL, tempLen, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if(add==MAP_FAILED) {
		perror("mmap()");
		printf("%s\n",strerror(errno));
		exit(1);
	}	
	*saveLen = tempLen;
	return add;
}


char getRandChar() {
	//liefert ein zufälliges druckbares Zeichen 
	/* Druckbare Zeichen beginnen bei 32 (A) und enden bei 126 (~) -> Spanne von 94 */
	const char start = 'A';
	const char end   = '~';
	const char range = end-start;
	char  c = 32 + getRandFloat()*range; 
	return c;
}

char * getRandString(size_t n) {
	// liefert eine zufällige Zeichenkette 
	char *str = (char *)malloc(n*sizeof(char)+1);
	if(str==NULL) {
		perror("malloc() failed in getRandString()");
		exit(1);
	}
	str[n] = '\0'; // Stringende 
	while(n--)
		str[n] = getRandChar();
	return str;
}

float getRandFloat() { // liefert eine Zufallszahl zwischen 0 und 1 (inklusive) 
	return ((float)rand())/RAND_MAX;
}
/*

#ifdef USE_CUDA
	__global__ void searchKernel(void *entrySet, long *result, int entries) {
		// implementiert memcmp auf Basis von <long> Vergleichen 
		long idx = threadIdx.x + blockIdx.x * blockDim.x;
		const long *c1,*c2;
		char n, diff; // diff: der aktuelle Thread soll nicht öfter laufen, als nötig (auf gesamten Kernelaufruf nicht ausweitbar) 
		char n_init = 32/sizeof(long); // 4
		while(idx<entries) { // Threads werden recycled, siehe Inkrement am Fuß der Schleife
			diff = 0; // FALSE
			n = n_init; // 4 Vergleiche 
			// Pointer jeweils auf den Anfang setzen 
			c1 = (long *)goldenHash;
			c2 = (long *)((journalentry *)entrySet)[idx].hash;
			while(n--) {
				if(*c1 != *c2) { // Abweichung
					diff = 1;
					break;
				}
				c1++;
				c2++;
			}
			if(!diff) { // treffer
				*result = idx; // Thread-Index ist die Nummer des Eintrags
				idx = entries; // dieser thread braucht nicht weitersuchen
			}
			idx += blockDim.x * gridDim.x; // aktueller index + (anzahl der Blöcke * Threads pro Block) 
		}
		//ein thread soll noch etwas anderes machen 
		
		//if(idx == (entries-1)) {
			// vielleicht Hash hinzufügen? 
		//}
		return;
	} 



	long isHashInJournalGPU(char *hash, void *haystack, off_t stacksize) {
		CUDA_HANDLE_ERR( cudaMemcpyToSymbol(goldenHash, hash, 32) ); // die gesuchte Prüfsumme wird in den Cache der GPU gebracht 
		long result = -1L; // nur der erfolgreiche Thread schreibt hier seine ID rein 
		searchKernel<<<blocks,threadsPerBlock>>>(haystack, &result, stacksize);
		return result;
	}

	__host__ void cudaCopyJournal(void *dev, void *host, off_t len) {
		CUDA_HANDLE_ERR( cudaMalloc((void**)&dev, len) ); // GPU Speicher wird alloziert
		CUDA_HANDLE_ERR( cudaMemcpy(dev, host, len, cudaMemcpyHostToDevice) ); // Datentransfer von Host Speicher nach VRAM 
	}

	__host__ void cudaExtendHashStack(void *add, void *entry, int offset) {
		CUDA_HANDLE_ERR( cudaMemcpy((void *)(((journalentry *)add)+offset), entry, sizeof(journalentry), cudaMemcpyHostToDevice) );
	}
#endif // USE_CUDA

*/