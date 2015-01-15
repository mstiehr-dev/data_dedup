/* mappedFileToVRAM.cu */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>


static void cudaCheckError(cudaError_t error, const char *file, int line) {
			if(error!=cudaSuccess) {
				printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
				exit(EXIT_FAILURE);
			}
		}
		#define CUDA_HANDLE_ERR(err) (cudaCheckError(err,__FILE__, __LINE__))

int main(int argc, char **argv) {
	char * filename = "/etc/fstab";
	FILE * f = fopen(filename, "rt");
	fseek(f,0,SEEK_END);
	long fsize = ftell(f); // Dateilänge ermitteln 
	fseek(f,0,SEEK_SET);
	char * add = (char *)mmap(NULL,fsize, PROT_READ, MAP_SHARED, fileno(f), 0); // Datei mappen
	
	void * dev;
	CUDA_HANDLE_ERR( cudaMalloc((void**)&dev,fsize) ); // VRAM reservieren 
	CUDA_HANDLE_ERR( cudaMemcpy(dev, add, fsize, cudaMemcpyHostToDevice) ); // VRAM füllen 
	
	char * buf = (char *) malloc(sizeof(char)*(fsize+1));  // Host-Puffer 
	strncpy(buf, add, fsize+1);
	printf("%s\n",buf); // Testausgabe der Originaldaten 
	printf("\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n\n");
	memset(buf,0,fsize+1); // Host-Puffer löschen 
	CUDA_HANDLE_ERR( cudaMemcpy(buf, dev, fsize, cudaMemcpyDeviceToHost) ); // Daten von VRAM auf Host-Puffer holen 
	printf("%s\n",buf); // erneute Ausgabe 
	
	// Aufräumen 
	CUDA_HANDLE_ERR( cudaFree(dev) );
	munmap(add, fsize);
	free(buf);
	return 0;
}
