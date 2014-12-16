/* findDataInGPUMem.cu */



#include "data_dedup.h" 

#define BLOCKS 10


char   randChar();
char * randString(size_t);
float  randFloat();

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

/*
__device__ int whatToCompare(char *s1, journalentry *s2, size_t n) {
	// Aufruf: whatToCompare((char *)resp, (journalentry *)s2, 32);
	// erfolgreicher zugriff auf den String im Journaleintrag mittels (char)(s2->hash)[n]
	while(n--) 
		s1[n] = (char)(s2->hash)[n];
	return 0;
}
__device__ int whatToCompare2(char *s1, journalentry *s2, size_t n) {
	while(n--) 
		s1[n] = (char)(s2->hash)[n];
	return 0;
}
__global__ void testkernel(void *s1, void *s2, void *resp, size_t n) {
	//whatToCompare((char *)resp, (journalentry *)s2, 32);
	whatToCompare2((char *)resp, (journalentry *)s1, 32);
	return; 
}
*/
__device__ int compareHashes(const char *s1, const char *s2, size_t n) {
	const char *c1=s1, *c2=s2;
	while(n--) {
		if(*c1!=*c2)
			return (*c2-*c1);
		c1++;
		c2++;
	}
	return 0;
}
__global__ void kernel(void *wantedEntry, void *entrySet, int *resp, int entries) {
	int i=0; 
	while(i<entries) {
		if(compareHashes(((journalentry *)wantedEntry)->hash, ((journalentry *)entrySet+i)->hash,32)==0) {
			// TREFFER
			*resp = i;
			return;
		}
		i++;
	}
	*resp = (-1);
	return;
} 



int main(int argc, char **argv) {
	// --- Initialisierungen
	srand(time(NULL));
	// --- lokalen Speicher bereitstellen und initialisieren 
	journalentry * host_data = (journalentry *) malloc(100*sizeof(journalentry));
	memset(host_data, 0, 100*sizeof(journalentry));
	int i;
	for(i=0; i<100; i++) {
		(host_data+i)->block = LONG_MAX * randFloat();
		(host_data+i)->len = SHRT_MAX * randFloat();
		char *tString = randString(32);
		strncpy((host_data+i)->hash,tString,32);
		free(tString);
	}
	// künstlich für Duplikat sorgen: 
	// memcpy(host_data+99, host_data, sizeof(journalentry));
	for(i=0; i<100; i++) // Testausgabe 
		printf("%ld -> %s -> %i\n",(host_data+i)->block, (host_data+i)->hash, (host_data+i)->len);
	

	// datensätze auf GPU bringen 
	void * dev_data;
	CUDA_HANDLE_ERR( cudaMalloc((void**)&dev_data, 100*sizeof(journalentry)) );
	CUDA_HANDLE_ERR( cudaMemcpy(dev_data, host_data, 100*sizeof(journalentry), cudaMemcpyHostToDevice) );
	
	
	void * dev_wantedEntry;
	journalentry findMe; 
	memcpy(&findMe, host_data+20, sizeof(journalentry));
	printf("so we're looking for this hash: [%s]\n", findMe.hash);
	CUDA_HANDLE_ERR( cudaMalloc((void**)&dev_wantedEntry, sizeof(journalentry)) );
	CUDA_HANDLE_ERR( cudaMemcpy(dev_wantedEntry, &findMe, sizeof(journalentry), cudaMemcpyHostToDevice) );
	
	// außerdem muss der Kernel irgendwo die Antwort speichern können: 
	int host_resp; 
	int *dev_resp; 
	CUDA_HANDLE_ERR( cudaMalloc((void**)&dev_resp, sizeof(int)) );
	
	kernel<<<1,1>>>(dev_wantedEntry, dev_data, dev_resp, 100);
	CUDA_HANDLE_ERR( cudaMemcpy(&host_resp, dev_resp, sizeof(int), cudaMemcpyDeviceToHost) );
	if(host_resp>=0) 
		printf("got your hash in tupel #%d!\n",host_resp);
	else
		printf("sorry pal - return value is %d\n", host_resp);
	cudaFree(dev_wantedEntry);
	cudaFree(dev_data);	
	cudaFree(dev_resp);
	free(host_data);
	/*
	testkernel<<<1,1>>>(dev_wantedEntry, dev_data, dev_resp, 100);
	char msg[33];
	CUDA_HANDLE_ERR( cudaMemcpy(&msg, dev_resp, 32, cudaMemcpyDeviceToHost) );
	msg[32] = '\0';
	printf("Antwort: %s\n",msg);
	*/
	return 0;
}


char randChar() {
	/* Druckbare Zeichen beginnen bei 32 (A) und enden bei 126 (~) -> Spanne von 94 */
	const char start = 'A';
	const char end   = '~';
	const char range = end-start;
	char  c = 32 + randFloat()*range; 
	return c;
}
char * randString(size_t n) {
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

float randFloat() {
	return ((float)rand())/RAND_MAX;
}
