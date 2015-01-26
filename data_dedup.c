/* data_dedup.c */

#ifdef USE_CUDA
	#include "data_dedup.cuh"
	//__constant__ char goldenHash[33];	// im Constant-Cache gehaltener Such-String
	//int blocks = 4;	// Konfiguration des Kernelaufrufs: Anzahl der Blöcke || beste Performance: 2* MultiProcessorCount
	//int threadsPerBlock = 1024; // maximum
#else
	#include "data_dedup.h"
#endif // USE_CUDA


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
