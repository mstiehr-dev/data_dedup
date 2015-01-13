/* data_dedup_cuda.cu */

#include "data_dedup.h";

__global__ void kernel(void *entrySet, long *result, int entries) {
	// implementiert memcmp auf Basis von <long> Vergleichen 
	long idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long *c1,*c2;
	char n, diff; // diff: der aktuelle Thread soll nicht öfter laufen, als nötig (auf gesamten Kernelaufruf nicht ausweitbar) 
	while(idx<entries) { // Threads werden recycled, siehe Inkrement am Fuß der Schleife
		diff = 0; // FALSE
		n=32/sizeof(long); // 4 Vergleiche 
		// Pointer jeweils auf den Anfang setzen 
		c1 = (long *)findMe;
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
	long result = -1;
	kernel<<1,1>>(haystack, &result, stacksize);
	return result;
}
