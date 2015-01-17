/* data_dedup_cuda.cu */

#include "data_dedup.h"
#include "data_dedup.cuh"



__global__ void searchKernel(void *entrySet, long *result, int entries) {
	// implementiert memcmp auf Basis von <long> Vergleichen 
	long idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long *c1,*c2;
	char n, diff; // diff: der aktuelle Thread soll nicht öfter laufen, als nötig (auf gesamten Kernelaufruf nicht ausweitbar) 
	while(idx<entries) { // Threads werden recycled, siehe Inkrement am Fuß der Schleife
		diff = 0; // FALSE
		n=32/sizeof(long); // 4 Vergleiche 
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
	return;
} 



__host__ long isHashInJournalGPU(char *hash, void *haystack, int stacksize) {
	CUDA_HANDLE_ERR( cudaMemcpyToSymbol(goldenHash, hash, 32) ); // die gesuchte Prüfsumme wird in den Cache der GPU gebracht 
	long result = -1L;
	searchKernel<<<blocks,threadsPerBlock>>>(haystack, &result, stacksize);
	return result;
}

__host__ void cudaCopyJournal(void *dev, void *host, off_t len) {
	CUDA_HANDLE_ERR( cudaMalloc((void**)&dev, len) ); // GPU Speicher wird alloziert
	CUDA_HANDLE_ERR( cudaMemcpy(dev, host, len, cudaMemcpyHostToDevice) ); // Datentransfer von Host Speicher nach VRAM 
}

__host__ void cudaExtendHashStack(void *add, journalentry *entry) {
	CUDA_HANDLE_ERR( cudaMemcpy(add, record, sizeof(journalentry), cudaMemcpyHostToDevice) );
}