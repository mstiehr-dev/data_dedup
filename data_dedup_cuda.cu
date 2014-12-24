/* data_dedup_cuda.cu */

#include "data_dedup.h";

__global__ void kernel(void *entrySet, long *result, int entries) {
	long idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long *c1,*c2;
	char n, diff;
	while(idx<entries) { 
		diff = 0;
		n=4;
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
			*result = idx;
			idx = entries; // dieser thread braucht nicht weitersuchen
		}
		idx += blockDim.x * gridDim.x;
	}
	return;
} 


__host__ long isHashInJournalGPU(char *hash, void *haystack, int stacksize) {
	CUDA_HANDLE_ERR( cudaMemcpyToSymbol(goldenHash, hash, 32) );
	long result = -1;
	kernel<<1,1>>(haystack, &result, stacksize);
	return result;
}