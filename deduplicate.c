/* deduplicate.c */

#ifdef USE_CUDA
	#include "data_dedup.cuh"
#else 
	#include "data_dedup.h"
#endif

//#define DEBUG

/* man könnte auch noch das Metafile binär gestalten und mmapen, alle x-tausend einträge vergrößern etc 
 * das würde den ständigen schreibaufwand verringern */
#ifdef USE_CUDA
	#include "data_dedup.cuh"
	__constant__ char goldenHash[33];	// im Constant-Cache gehaltener Such-String
	int blocks = 4;	// Konfiguration des Kernelaufrufs: Anzahl der Blöcke || beste Performance: 2* MultiProcessorCount
	int threadsPerBlock = 1024; // maximum
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
		/* ein thread soll noch etwas anderes machen */ 
		/*
		if(idx == (entries-1)) {
			// vielleicht Hash hinzufügen? 
		} */
		return;
	} 
#endif // USE_CUDA

cudaDeviceProp prop; // zur Ermittlung der GPU Eckdaten 
size_t totalGlobalMem; 
size_t sharedMemPerBlock;
int max_threadsPerBlock;

void cudaCopyJournal(void *, void *, off_t);
void cudaExtendHashStack(void *, void *, int);

time_t startZeit;
double laufZeit;

int main(int argc, char **argv) {
	char *inputFileName = NULL;
	int c = 0;
	// parse command line arguments:
	opterr = 0;
	while((c=getopt(argc, argv, "?f:r"))!=-1) { // : -> argument required
		switch(c) {
			case 'h': 	printf("usage: %s -f <filename>\n", *argv); break;
			case 'f':	if(optarg) inputFileName = optarg; break;
			case '?':	printf("usage: %s -f <filename>\n", *argv); break;
			case 'r':	// reassemblieren 
						printf("this feature is not yet implemented. please refer to \"./reassemble\" instead.\n");
						exit(1);
			default:	printf("this option is currently not supported. please see '-h'.\n");
						exit(1);
		}
	}
	if(inputFileName==NULL) {
		printf("ERROR: no input file given - QUIT\n");
		printf("See %s -h for help!\n",argv[0]);
		exit(1);
	}
	
	// DIE ZU DEDUPLIZIERENDE DATEI
	FILE *inputFile = fopen(inputFileName, "rb");
	if(inputFile==NULL) {
		fprintf(stderr,"ERROR: could not open %s!\n"
					   "Please check for existence and proper permissions\n",inputFileName);
		perror("fopen()");
		exit(1);
	}
	struct stat inputFileStats; // Dateiattribute ermitteln 
	if(fstat(fileno(inputFile),&inputFileStats)==-1) {
		perror("fstat()");
		exit(1);
	}
	off_t inputFileLen = inputFileStats.st_size;
	double inputFileLenMB = inputFileLen/(1024.0*1024.0);
	
	
	// DIE INDEXDATEI ÜBER ALLE HASHES (JOURNAL)
	FILE *journalFile = fopen(JOURNALFILE,"a+b");
	if(journalFile==NULL) {
		fprintf(stderr,"ERROR: could not open %s!\n",JOURNALFILE);
		perror("fopen()");
		exit(1);
	}
	struct stat journalFileStats; // Attribute des Journals ermitteln
	if(fstat(fileno(journalFile),&journalFileStats)==-1) {
		perror("fstat()");
		exit(1);
	}
	off_t journalFileLen = journalFileStats.st_size;
	off_t journalEntries = journalFileLen / sizeof(journalentry);	
	off_t journalMapLen = 0L;
	// Journal mappen + Platz für 100 Einträge 
	void *journalMapAdd = mapFile(fileno(journalFile),journalFileLen, auxSpace, &journalMapLen);
	void *journalMapCurrentEnd = ((char *)journalMapAdd) + journalFileLen; // Hilfszeiger soll ans Dateiende zeigen 
	#ifdef DEBUG
		printf("JournalFileLen: %ld\n", journalFileLen);
		printf("JournalEntries: %ld\n", journalEntries);
		printf("JournalMapLen : %ld\n", journalMapLen);
		printf("JournalMapAdd : %p\n", journalMapAdd);
	#endif
	
	// STELLVERTRETER FÜR DIE DEDUPLIZIERTE DATEI (METAFILE) 
	char *metaFileName = (char *)buildString3s(METADIR,basename(inputFileName), ".meta");
	FILE *metaFile = fopen(metaFileName,"rb");
	if(metaFile!=NULL) { // Datei existiert bereits 
		fprintf(stdout,"WARNING: file \'%s\' already exists.\nOverwrite? (y/N) >",metaFileName);
		char stdinBuf[2];
		fgets(stdinBuf,2,stdin);
		if('Y' != (*stdinBuf&0x59)) {// nicht überschreiben -> Abbruch
			fprintf(stdout,"=> Abbruch\n");
			fcloseall(); // alle filedeskriptoren schließen 
			if(metaFileName) free(metaFileName);
			return(1);
		}
		/* es soll überschrieben werden */
		fclose(metaFile); // es wird erneut geöffnet (im Schreibmodus) 
	}
	metaFile = fopen(metaFileName,"wb"); // nur schreiben, falls existent, löschen 
	if(metaFile==NULL) {
		fprintf(stderr,"ERROR: could not open %s!\n",metaFileName);
		perror("fopen()");
		exit(1);
	}
	
	
	// DATENHALDE ÖFFNEN
	FILE *storageFile = fopen(STORAGEDUMP, "a+b");
	if(storageFile==NULL) {
		fprintf(stderr,"ERROR: could not open %s!\n",STORAGEDUMP);
		perror("ERROR: could not open storage dump file");
		exit(1);
	}
	fseek(storageFile,0,SEEK_END); // ans Ende spulen (dort werden Daten angehängt 
	struct stat storageFileStats; // Dateiattribute des Dumps ermitteln
	if(fstat(fileno(storageFile),&storageFileStats)==-1) {
		perror("fstat()");
		exit(1);
	}
	off_t storageFileLen = storageFileStats.st_size;
	
// #### VERARBEITUNG AUF GPU
#ifdef USE_CUDA
	// Journaldaten in VRAM geben
	if(journalMapLen> 1027604480) { // gemessen 
		// Datenmenge übersteigt Grafikspeicher 
		printf( "Das Datenvolumen übersteigt die Speicherkapazität der Grafikkarte.\n"
				"In der aktuellen Version wird dieser Fall werden maximal 25.675 mio. Datensätze unterstützt. Beende.\n");
		fcloseall();
		exit(2);
	}
	void * VRAM; // Adresse des Grafikspeichers
	CUDA_HANDLE_ERR( cudaMalloc((void**)&VRAM, journalMapLen) ); // GPU Speicher wird alloziert
	CUDA_HANDLE_ERR( cudaMemcpy(VRAM, journalMapAdd, journalMapLen, cudaMemcpyHostToDevice) ); // Datentransfer von Host Speicher nach VRAM 
	//cudaCopyJournal(VRAM, journalMapAdd, journalMapLen); // auch im VRAM wird ein zusätzlicher Puffer reserviert (siehe journalMapLen)
#endif

// BEGINN DER VERARBEITUNG
	startZeit = time(NULL);	
	long newBytes = 0; // Blöcke, die neu ins Journal aufgenommen wurden 
	char * inputFileBuffer; // dort wird die Datei in Stückchen gepuffert 
	// DIE EINGABEDATEI EINLESEN
	off_t bytesActuallyBuffered = 0L;
	off_t bytesBufferedTotal = 0L;
	printf("deduplicating \"%s\" [%.3f MB]\n",inputFileName, inputFileLenMB);
	char *md5String = (char *) NULL;
	// Die Schleife verarbeitet die Eingabedatei in Schritten von <bytesBufferSize> Byte, bis die gesamte Datei gelesen wurde 
	const unsigned int bytesBufferSize = 8*1024*1024; // 128 MB
	off_t progress = 0, delta = 0;
	time_t start = time(NULL);
	for(bytesBufferedTotal = 0L; bytesBufferedTotal<inputFileLen; bytesBufferedTotal+=bytesActuallyBuffered) {
		
		inputFileBuffer = (char *) malloc(sizeof(char)*bytesBufferSize);
		if(inputFileBuffer==NULL) {
			printf("ERROR: could not allocate %i bytes of memory",bytesBufferSize);
			perror("malloc");
			fcloseall();
			exit(1);
		}
		bytesActuallyBuffered = fread(inputFileBuffer,1,bytesBufferSize,inputFile); // liest 1*bytesBufferSize aus inputFile nach inputFileBuffer (Rückgabewert ist Anzahl der gelesenen Bytes, wenn size 1 ist )
		//bytesActuallyBuffered = (inputFileLen-bytesBufferedTotal >= bytesBufferSize) ? bytesBufferSize : (inputFileLen-bytesBufferedTotal); // ermitteln, wie viel gelesen wurde 
		
		// FÜR JEDEN CHUNK EINEN HASH BERECHNEN UND MIT DEM JOURNAL VERGLEICHEN 
		md5String = (char *) malloc(sizeof(char)*(32+1));
		if(md5String==NULL) {
			perror("ERROR: could not allocate 33 bytes of memory for md5String\n");
			fcloseall();
			exit(1);
		}
		int current_read = 0; // wie viele Bytes aktuell vorhanden sind 
		size_t bytesRead; // Summe der konsumierten Bytes 
		char journalFileChanged; // Flag über Modifizierung des Journalfiles 
		long infoForMetaFile = -1L; // enthält die jeweilige Zeilennummer des Journals
		MD5_CTX md5Context;   // Struktur für die Hash-Berechnung
		unsigned char md[16];
		// Schleife geht in Schritten von 512 Byte über den aktuellen Puffer, berechnet jeweils den Hash und prüft, ob dieser bereits existiert 
		for(bytesRead=0; bytesRead<bytesActuallyBuffered; ) {
			journalFileChanged = FALSE;
			current_read = (CHUNKSIZE<=(bytesActuallyBuffered-bytesRead)) ? CHUNKSIZE : bytesActuallyBuffered-bytesRead;
			if(MD5_Init(&md5Context)==0) { // 1 == success, 0 == fail
				perror("MD5_Init()");
				fcloseall();
				exit(1);
			}
			MD5_Update(&md5Context, inputFileBuffer+bytesRead, current_read); // hash berechnen 
			MD5_Final(md, &md5Context); // hash in md[16] speichern
			int i;
			for(i=0;i<16;i++)  // Hash-String bauen 
				sprintf(md5String+2*i, "%02x", (unsigned int) md[i]);

// #### HASH SUCHE 
			long hashInJournalPos = -1L;
#ifndef USE_CUDA
			hashInJournalPos = isHashInMappedJournal(md5String, journalMapAdd, journalEntries);
#else 
			//hashInJournalPos = isHashInJournalGPU(md5String, VRAM, journalEntries);
			CUDA_HANDLE_ERR( cudaMemcpyToSymbol(goldenHash, md5String, 32) ); // die gesuchte Prüfsumme wird in den Cache der GPU gebracht 
			long result = -1L; // nur der erfolgreiche Thread schreibt hier seine ID rein 
			searchKernel<<<1,1>>>(journalMapAdd, &result, journalEntries);
			hashInJournalPos = result;
#endif
			if(hashInJournalPos==-1) { // DER HASH IST UNBEKANNT -> MUSS ANGEFÜGT WERDEN 
				//printf("+"); //fflush(stdout);
				infoForMetaFile = journalEntries++; // in diesem Datensatz wird sich der neue Hash befinden
				journalentry record; // neuen Eintrag bauen 
				record.block = storageFileLen; // ganz hinten anfügen -> aktuelles Dateiende
				strncpy(record.hash, md5String, 32+1); // die Prüfsumme wird übernommen
				record.len = current_read; // die Blocklänge 
				fwrite(inputFileBuffer+bytesRead, current_read, 1, storageFile); // Daten an Dump anfügen
				memcpy(journalMapCurrentEnd, &record, sizeof(journalentry)); // Eintrag im Journal vornehmen 
				#ifdef DEBUG
					printf("\n%ld -> %s -> %d\n", record.block, record.hash, record.len);
					printf("journalMapCurrentEnd: %p\n", journalMapCurrentEnd);
					printf("return memcpy       : %p\n", ret);
				#endif
			#ifdef USE_CUDA
				// auch der Datenbestand im Videospeicher muss erweitert werden 
					void *t = malloc(sizeof(journalentry));
					memcpy(t,&record, sizeof(record));
					CUDA_HANDLE_ERR( cudaMemcpy((void *)(((journalentry *)VRAM)+journalEntries), t, sizeof(journalentry), cudaMemcpyHostToDevice) );

				//cudaExtendHashStack(VRAM,t, (int)journalEntries);
					free(t);
			#endif	
				journalMapCurrentEnd = ((journalentry *)journalMapCurrentEnd) + 1; // neues Journal-Ende 
				journalFileChanged = TRUE;
				if(journalEntries*sizeof(journalentry) >= journalMapLen) {
				// die Journal-Datei muss vergrößert und erneut gemappt werden 
					munmap(journalMapAdd, journalMapLen); // synchronisiert mit Dateisystem 
					journalMapAdd = mapFile(fileno(journalFile),journalMapLen, auxSpace, &journalMapLen); // remap 
					journalMapCurrentEnd = ((journalentry*)journalMapAdd) + journalEntries;
				// auch der VRAM muss aktualisiert werden: 
				#ifdef USE_CUDA
					CUDA_HANDLE_ERR( cudaFree(VRAM) );
					CUDA_HANDLE_ERR( cudaMalloc((void**)&VRAM, journalMapLen) ); // GPU Speicher wird alloziert
					CUDA_HANDLE_ERR( cudaMemcpy(VRAM, journalMapAdd, journalMapLen, cudaMemcpyHostToDevice) ); // Datentransfer von Host Speicher nach VRAM 
	
					//cudaCopyJournal(VRAM, journalMapAdd, journalMapLen);
				#endif
					laufZeit = difftime(time(NULL),start);
					delta = progress;
					progress = bytesBufferedTotal+bytesRead + current_read;
					delta = progress - delta;
					if(laufZeit>0.5) {
						printf("\n+++++++++++++++++++++++++++++++++++++++++++\n");
						printf("Fortschritt: %3.2f%%\n", (progress*100.0)/inputFileLen);
						double speed = (delta/(1024.0*1024.0))/laufZeit; // in MB/s
						printf("aktuelle Geschwindigkeit: %.3f MB/s\n", speed);
						//printf("delta: %.2f MByte, zeit: %.1fs\n", delta/(1024.0*1024.0), laufZeit);
						printf("verbleibend: %.1f MB [~%.1f s]\n", (inputFileLen-progress)/(1024.0*1024.0), (inputFileLen-progress)/(speed*1024.0*1024.0));
						printf("+++++++++++++++++++++++++++++++++++++++++++\n");
					}
					start = time(NULL);
				}
			} else { // DER HASH IST BEREITS BEKANNT
				//printf("."); //fflush(stdout);
				infoForMetaFile = hashInJournalPos; // die zeile des journals, in der der hash gefunden wurde, wird ins metafile übernommen 
			}
			// Informationen ins Metafile schreiben
			#ifdef DEBUG 
			printf("Schreibe in Metafile: %ld\n", infoForMetaFile);
			#endif
			fwrite(&infoForMetaFile, sizeof(infoForMetaFile), 1, metaFile);
			//fprintf(metaFile, "%ld\n", infoForMetaFile);
			if(journalFileChanged) {
				newBytes += current_read;
				storageFileLen += current_read;
			}
			bytesRead += current_read;
		}
		if(inputFileBuffer) free(inputFileBuffer);
		if(md5String) 		free(md5String);
	}
	// Nachbereitung
	laufZeit = difftime(time(NULL),startZeit);
	if(laufZeit<0.01) laufZeit=0.01;
	double speed = inputFileLenMB/laufZeit;
	if((munmap(journalMapAdd, journalMapLen))==-1) {
		perror("munmap");
		exit(1);
	}
	/* Journal-Datei wieder verkleinern */
	if(ftruncate(fileno(journalFile),journalEntries*sizeof(journalentry))==-1) {
		perror("ftruncate()");
		exit(1);
	}
	if(metaFileName) free(metaFileName);
	fcloseall();
#ifdef USE_CUDA 
	/* der VRAM muss freigegeben werden */ 
	CUDA_HANDLE_ERR( cudaFree(VRAM) );
#endif
	printf("\n\n*** successfully deduplicated \"%s\" in %.1fs [%.3f MB/s] ***\n", inputFileName, laufZeit, speed);
	printf("*** added %ld Bytes to storage dump ***\n",newBytes);
	return 0;
}
