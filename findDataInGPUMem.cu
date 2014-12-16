/* findDataInGPUMem.cu */



#include "data_dedup.h" 

#ifndef HAYSTACK
  #define HAYSTACK 1000
#endif
#ifndef BLOCKS
  #define BLOCKS 4
#endif
#ifndef THREADS
  #define THREADS 256
#endif


__constant__ journalentry findMe[1];

/*
__global__ void findHash(void *mem, int sets, void *e) {
	int block = blockIdx.x;
	int slice = sets / BLOCKS;
	int i;
	for(i=block*slice; i<(block+1)*slice; i++) {
		if( memcmp(((journalentry*)mem)+i*sizeof(journalentry), (journalentry *) e, sizeof(journalentry)) == 0) {
			return; // TREFFER 
		}
	}
	return;
}
*/

__device__ int compareHashes(/*const char *s1, */const char *s2, size_t n) {
	const char *c1=findMe[0].hash, *c2=s2;
	while(n--) {
		if(*c1!=*c2)
			return (*c2-*c1);
		c1++;
		c2++;
	}
	return 0;
}
__global__ void kernel(/*void *wantedEntry, */void *entrySet, int *resp, int entries) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx<HAYSTACK) { // den wantedhash irgendwo cachen!!! 
		if(compareHashes( /*((journalentry *)wantedEntry)->hash, */((journalentry *)entrySet+idx)->hash,32) == 0 ) {
			// Treffer -> alle anderen können aufhören
			*resp = idx; // wird ständig überschrieben, weil kernel auch nach treffer nicht terminiert
			//asm("trap;"); // ekliger abbruch, führt zu fehlern
			return;
		}
		idx += blockDim.x * gridDim.x;
	}
	// wenn wir hier ankommen, gab es keinen Treffer -> Fehlercode -1
	//*resp = (-1);
	return;
} 




int main(int argc, char **argv) {
	srand(time(NULL));
	unsigned int treffer = randFloat() * HAYSTACK; // Dieser Datensatz wird nachher im HAYSTACK gesucht
	cudaEvent_t start, stop; 
	float elapsedTime;
	// --- lokalen Speicher bereitstellen und initialisieren 
	journalentry * host_data = (journalentry *) malloc(HAYSTACK*sizeof(journalentry));
	memset(host_data, 0, HAYSTACK*sizeof(journalentry));
	int i;
	for(i=0; i<HAYSTACK; i++) {
		(host_data+i)->block = LONG_MAX * randFloat();
		(host_data+i)->len = SHRT_MAX * randFloat();
		char *tString = randString(32);
		strncpy((host_data+i)->hash,tString,32);
		free(tString);
	}
	// künstlich für Duplikat sorgen: 
	// memcpy(host_data+99, host_data, sizeof(journalentry));
	//for(i=0; i<HAYSTACK; i++) // Testausgabe 
	//	printf("%ld -> %s -> %i\n",(host_data+i)->block, (host_data+i)->hash, (host_data+i)->len);

	CUDA_HANDLE_ERR( cudaEventCreate(&start)   );
	CUDA_HANDLE_ERR( cudaEventCreate(&stop)    );
	CUDA_HANDLE_ERR( cudaEventRecord(start, 0) );
	
	// datensätze auf GPU bringen 
	void * dev_data;
	CUDA_HANDLE_ERR( cudaMalloc((void**)&dev_data, HAYSTACK*sizeof(journalentry)) );
	CUDA_HANDLE_ERR( cudaMemcpy(dev_data, host_data, HAYSTACK*sizeof(journalentry), cudaMemcpyHostToDevice) );
	
	
	//void * dev_wantedEntry; 
	CUDA_HANDLE_ERR( cudaMemcpyToSymbol(findMe, host_data+treffer, sizeof(journalentry)) );
	//memcpy(&findMe, host_data+treffer, sizeof(journalentry));
	printf("so we're looking for this hash: [%s]\n", host_data[treffer].hash);
	//CUDA_HANDLE_ERR( cudaMalloc((void**)&dev_wantedEntry, sizeof(journalentry)) );
	//CUDA_HANDLE_ERR( cudaMemcpy(dev_wantedEntry, &findMe, sizeof(journalentry), cudaMemcpyHostToDevice) );
	
	// außerdem muss der Kernel irgendwo die Antwort speichern können: 
	int host_resp=-1; 
	int *dev_resp; 
	CUDA_HANDLE_ERR( cudaMalloc((void**)&dev_resp, sizeof(int)) );
	CUDA_HANDLE_ERR( cudaMemcpy(dev_resp, &host_resp, sizeof(int), cudaMemcpyHostToDevice) );
	kernel<<<BLOCKS,THREADS>>>(/*dev_wantedEntry, */dev_data, dev_resp, HAYSTACK);
	CUDA_HANDLE_ERR( cudaMemcpy(&host_resp, dev_resp, sizeof(int), cudaMemcpyDeviceToHost) );
	CUDA_HANDLE_ERR( cudaEventRecord(stop,0) );
	CUDA_HANDLE_ERR( cudaEventSynchronize(stop) );
	CUDA_HANDLE_ERR( cudaEventElapsedTime(&elapsedTime, start, stop) );
	if(host_resp>=0) 
		printf("got your hash in tupel #%d!\n",host_resp);
	else
		printf("sorry pal - return value is %d\n", host_resp);
	printf("### computation took %fms\n",elapsedTime);
	printf("### Using %d Threads in a (%dx%d) Grid\n", (BLOCKS*THREADS), BLOCKS, THREADS);
	printf("### Haystack: %d\n",HAYSTACK);
	//CUDA_HANDLE_ERR( cudaFree(dev_wantedEntry) );
	CUDA_HANDLE_ERR( cudaFree(dev_data) );
	CUDA_HANDLE_ERR( cudaFree(dev_resp) );
	CUDA_HANDLE_ERR( cudaEventDestroy(start) );
	CUDA_HANDLE_ERR( cudaEventDestroy(stop) );
	free(host_data);
	
	return 0;
}








char randChar() {
	//liefert ein zufälliges druckbares Zeichen 
	/* Druckbare Zeichen beginnen bei 32 (A) und enden bei 126 (~) -> Spanne von 94 */
	const char start = 'A';
	const char end   = '~';
	const char range = end-start;
	char  c = 32 + randFloat()*range; 
	return c;
}
char * randString(size_t n) {
	// liefert eine zufällige Zeichenkette 
	char *str = (char *)malloc(n*sizeof(char)+1);
	if(str==NULL) {
		perror("malloc() failed in randString()");
		exit(1);
	}
	str[n] = '\0'; // Stringende 
	while(n--)
		str[n] = randChar();
	return str;
}

float randFloat() { // liefert eine Zufallszahl zwischen 0 und 1 (inklusive) 
	return ((float)rand())/RAND_MAX;
}
