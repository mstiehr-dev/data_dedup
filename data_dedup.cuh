


#ifndef CUDA_CUH
	#define CUDA_CUH

	#include "data_dedup.h"

	#ifdef USE_CUDA
		static void cudaCheckError(cudaError_t error, const char *file, int line) {
			if(error!=cudaSuccess) {
				printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
				exit(EXIT_FAILURE);
			}
		}
		#define CUDA_HANDLE_ERR(err) (cudaCheckError(err,__FILE__, __LINE__))
		cudaDeviceProp prop;
		size_t totalGlobalMem; 
		size_t sharedMemPerBlock;
		int max_threadsPerBlock;
		__constant__ char goldenHash[33];
		int blocks = 4;
		int threadsPerBlock = 256;
	#endif
#endif

