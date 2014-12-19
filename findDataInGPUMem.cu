/* findDataInGPUMem.cu */



#include "data_dedup.h" 
#include "unistd.h" //getopt

/* globale variablen: */
int _haystack = 10; // wird durch Command Line Parameter ersetzt 
int _blocks   = 1;  // wird durch Command Line Parameter ersetzt 
int _threads  = 1;  // wird durch Command Line Parameter ersetzt 
__constant__ char findMe[32];
/* hier wird der gesuchte Hash gespeichert im Constant Memory der GPU gecached */

// GPU Funktionen
/*
__device__ int compareHashesAsLong(const void *s2, size_t n) {
	// die Strings sind 32 Byte lang, deswegen bietet sich der Einsatz von 
	// long (8 Byte) an
	const long *c1=(long*)(findMe[0].hash), *c2=(long*)s2;
	n = (n+sizeof(long)-1)/sizeof(long);
	while(n--) {
		if(*c1!=*c2)
			return (-1); // Differenz ist hier egal
		c1++;
		c2++;
	}
	return 0;
}
__global__ void kernel(void *entrySet, int *result, int entries) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx<entries) { 
		if(compareHashesAsLong( ((journalentry *)entrySet+idx)->hash,32) == 0 ) {
			// Treffer
			*result = idx;
			//asm("trap;"); // harter abbruch des Kernels, führt zu Fehlern
			return;
		}
		idx += blockDim.x * gridDim.x;
	}
	return;
} 
*/
__global__ void kernel(void *entrySet, int *result, int entries) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const long *c1;
	const long *c2;
	char n;
	char diff;
	while(idx<entries) { 
		diff = 0;
		n=4;
		c1 = (long *)findMe;
		c2 = (long *)((journalentry *)entrySet[idx]).hash;
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


int main(int argc, char **argv) {
/* Vorbereitungen */
	// Command Line Arguments parsen: 
	int c;
	opterr = 0;
	while((c=getopt(argc, argv, "b:t:h:"))!=-1) { // : -> argument required
		switch(c) {
			//case 'h':	_haystack = atoi(optarg); break;
			case 'b':	if(optarg) _blocks   = atoi(optarg); break;
			case 't':	if(optarg) _threads  = atoi(optarg); break;
			case 'h':	if(optarg) _haystack = atoi(optarg); break;
			default:
				printf("usage: %s -h <size of _haystack> -b <_blocks> -t <_threads per block>\n",argv[0]);
				exit(1);
		}
	}
	srand(time(NULL));
	unsigned int treffer = getRandFloat() * _haystack; // Dieser Datensatz wird nachher im _haystack gesucht
	cudaEvent_t start, stop; 
	float elapsedTime;
	// wieviel Speicher hat die GPU? 
	cudaDeviceProp gpu; 
	CUDA_HANDLE_ERR( cudaGetDeviceProperties(&gpu, 0) );
	size_t totalGPUMem = gpu.totalGlobalMem;
	if(_haystack*sizeof(journalentry)>=totalGPUMem) {
		char user[2]; // 1 Buchstabe + \n
		printf("+++ WARNING +++\n");
		printf("+++ Memory usage exceeds GPU capacity!\n");
		printf("+++ continue? (y/N)\n");
		printf(" > ");
		fgets(user, 2, stdin);
		if('Y'!=(*user&0x59)) {
			printf("exit");
			exit(1);
		}
	}
	// --- lokalen Speicher bereitstellen und initialisieren 
	journalentry * host_data = (journalentry *) malloc(_haystack*sizeof(journalentry));
	memset(host_data, 0, _haystack*sizeof(journalentry));
	int i;
	for(i=0; i<_haystack; i++) {
		(host_data+i)->block = LONG_MAX * getRandFloat();
		(host_data+i)->len = SHRT_MAX * getRandFloat();
		char *tString = getRandString(32);
		strncpy((host_data+i)->hash,tString,32);
		free(tString);
	}

	CUDA_HANDLE_ERR( cudaEventCreate(&start)   );
	CUDA_HANDLE_ERR( cudaEventCreate(&stop)    );
	CUDA_HANDLE_ERR( cudaEventRecord(start, 0) );
	
	// datensätze auf GPU bringen 
	void * dev_data;
	CUDA_HANDLE_ERR( cudaMalloc((void**)&dev_data, _haystack*sizeof(journalentry)) );
	CUDA_HANDLE_ERR( cudaMemcpy(dev_data, host_data, _haystack*sizeof(journalentry), cudaMemcpyHostToDevice) );
	
	
	//void * dev_wantedEntry; 
	CUDA_HANDLE_ERR( cudaMemcpyToSymbol(findMe, (host_data+treffer).hash, 32) );
	//memcpy(&findMe, host_data+treffer, sizeof(journalentry));
	printf("so we're looking for this hash: [%s]\n", host_data[treffer].hash);
	//CUDA_HANDLE_ERR( cudaMalloc((void**)&dev_wantedEntry, sizeof(journalentry)) );
	//CUDA_HANDLE_ERR( cudaMemcpy(dev_wantedEntry, &findMe, sizeof(journalentry), cudaMemcpyHostToDevice) );
	
	// außerdem muss der Kernel irgendwo die Antwort speichern können: 
	int host_result=-1; 
	int *dev_result; 
	CUDA_HANDLE_ERR( cudaMalloc((void**)&dev_result, sizeof(int)) );
	CUDA_HANDLE_ERR( cudaMemcpy(dev_result, &host_result, sizeof(int), cudaMemcpyHostToDevice) );
	
	kernel<<<_blocks,_threads>>>(dev_data, dev_result, _haystack);
	
	CUDA_HANDLE_ERR( cudaMemcpy(&host_result, dev_result, sizeof(int), cudaMemcpyDeviceToHost) );
	CUDA_HANDLE_ERR( cudaEventRecord(stop,0) );
	CUDA_HANDLE_ERR( cudaEventSynchronize(stop) );
	CUDA_HANDLE_ERR( cudaEventElapsedTime(&elapsedTime, start, stop) );
	if(host_result>=0) 
		printf("got your hash in tupel #%d!\n",host_result);
	else
		printf("sorry pal - return value is %d\n", host_result);
	printf("\tcomputation took %fms\n",elapsedTime);
	printf("\tUsing %d _threads in a (%dx%d) Grid\n", (_blocks*_threads), _blocks, _threads);
	printf("\tHeuhaufen: %d Datensätze\n",_haystack);
	//CUDA_HANDLE_ERR( cudaFree(dev_wantedEntry) );
	CUDA_HANDLE_ERR( cudaFree(dev_data) );
	CUDA_HANDLE_ERR( cudaFree(dev_result) );
	CUDA_HANDLE_ERR( cudaEventDestroy(start) );
	CUDA_HANDLE_ERR( cudaEventDestroy(stop) );
	free(host_data);
	
	return 0;
}

// ######################################################################################

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
