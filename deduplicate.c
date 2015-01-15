/* deduplicate.c */


#include "data_dedup.h"


int main(int argc, char **argv) {
	char *inputFileName = NULL;
	int c = 0;
	// parse command line arguments:
	opterr = 0;
	while((c=getopt(argc, argv, "?f:"))!=-1) { // : -> argument required
		switch(c) {
			case 'f':	if(optarg) inputFileName = optarg; break;
			case '?':	printf("usage: %s -f <filename>\n", *argv); break;
			default:	break;
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
	
	
	// STELLVERTRETER FÜR DIE DEDUPLIZIERTE DATEI (METAFILE) 
	char *metaFileName = (char *)buildString3s(METADIR,basename(inputFileName), ".meta");
	FILE *metaFile = fopen(metaFileName,"rt");
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
	metaFile = fopen(metaFileName,"wt"); // nur schreiben, falls existent, löschen 
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
	
	
	
// #### VERARBEITUNG AUF DEM HOST
	// DAS JOURNAL MAPPEN (zusätzlicher Platz für 100 Einträge)
	off_t journalMapLen = 0L;
	void *journalMapAdd = mapFile(fileno(journalFile),journalFileLen, auxSpace, &journalMapLen);
	void *journalMapCurrentEnd = journalMapAdd + journalFileLen; // Hilfszeiger soll ans Dateiende zeigen 
	// STATISTIK
#ifdef USE_CUDA
	// Journaldaten in VRAM geben
	if(journalFileLen> 25675000) {
		// Datenmenge übersteigt Grafikspeicher 
		printf( "Das Datenvolumen übersteigt den Grafikspeicher.\n"
				"In der aktuellen Version wird dieser Fall werden maximal 25.675 mio. Datensätze unterstützt. Beende.\n");
		fcloseall();
		exit(2);
	}
	void * VRAM; 
	CUDA_HANDLE_ERR( cudaMalloc((void**)&VRAM, journalFileLen) );
#endif

// BEGINN DER VERARBEITUNG
startZeit = time(NULL);	
long newBytes = 0; // Blöcke, die neu ins Journal aufgenommen wurden 
char * inputFileBuffer; // dort wird die Datei in Stückchen gepuffert 
	// DIE EINGABEDATEI EINLESEN
	unsigned int bytesBufferSize = 10*1024*1024; // 10 MB
	off_t bytesActuallyBuffered = 0L;
	off_t bytesBufferedTotal = 0L;
	printf("deduplicating \"%s\" [%.3f MB]\n",inputFileName, inputFileLenMB);
	char *md5String = (char *) NULL;
	// Die Schleife verarbeitet die Eingabedatei in Schritten von <bytesBufferSize> Byte, bis die gesamte Datei gelesen wurde 
	for(bytesBufferedTotal = 0L; bytesBufferedTotal<inputFileLen; bytesBufferedTotal+=bytesActuallyBuffered) {
		inputFileBuffer = malloc(sizeof(char)*bytesBufferSize);
		if(inputFileBuffer==NULL) {
			printf("ERROR: could not allocate %i bytes of memory",bytesBufferSize);
			perror("malloc");
			fcloseall();
			exit(1);
		}
		fread(inputFileBuffer,1,bytesBufferSize,inputFile); // liest 1*bytesBufferSize aus inputFile nach inputFileBuffer (Rückgabewert ist Anzahl der gelesenen Bytes, wenn size 1 ist )
		bytesActuallyBuffered = (inputFileLen-bytesBufferedTotal >= bytesBufferSize) ? bytesBufferSize : (inputFileLen-bytesBufferedTotal); // ermitteln, wie viel gelesen wurde 
		
		// FÜR JEDEN CHUNK EINEN HASH BERECHNEN UND MIT DEM JOURNAL VERGLEICHEN 
		md5String = malloc(sizeof(char)*(32+1));
		if(md5String==NULL) {
			perror("ERROR: could not allocate 33 bytes of memory for md5String\n");
			fcloseall();
			exit(1);
		}
		int current_read = 0; // wie viele Bytes aktuell vorhanden sind 
		size_t bytesRead; // Summe der konsumierten Bytes 
		char metaFileChanged; // Flag über Modifizierung des Metafiles 
		long infoForMetaFile = -1; // enthält die jeweilige Zeilennummer des Journals
		MD5_CTX md5Context;   // Struktur für die Hash-Berechnung
		unsigned char md[16];
		// Schleife geht in Schritten von 512 Byte über den aktuellen Puffer, berechnet jeweils den Hash und prüft, ob dieser bereits existiert 
		for(bytesRead=0; bytesRead<bytesActuallyBuffered; ) {
			metaFileChanged = FALSE;
			current_read = (CHUNKSIZE<=(bytesActuallyBuffered-bytesRead)) ? CHUNKSIZE : bytesActuallyBuffered-bytesRead;
			if(MD5_Init(&md5Context)==0) { // 1 == success, 0 == fail
				perror("MD5_Init()");
				fcloseall();
				exit(1);
			}
			MD5_Update(&md5Context, inputFileBuffer+bytesRead, current_read); // hash berechnen 
			MD5_Final(md, &md5Context); // hash in md[16] speichern
			int i;
			for(i=0;i<16;i++)  // String bauen 
				sprintf(md5String+2*i, "%02x", (unsigned int) md[i]);	
// Testen, ob der errechnete Hash bereits bekannt ist
// #### HASH SUCHE 
			long hashInJournalPos = -1L;
#ifndef USE_CUDA
			hashInJournalPos = isHashInMappedJournal(md5String, journalMapAdd, journalEntries);
#else 
			// Journal in VRAM bringen 

			hashInJournalPos = isHashInJournalGPU(md5String, VRAM, journalEntries);
#endif
			if(hashInJournalPos==-1) { // DER HASH IST UNBEKANNT -> MUSS ANGEFÜGT WERDEN 
	printf("+"); fflush(stdout);
				infoForMetaFile = journalEntries++; // in diesem Datensatz wird sich der neue Hash befinden
				journalentry record; // neuen Eintrag bauen 
				record.block = storageFileLen; // ganz hinten anfügen -> aktuelles Dateiende
				strncpy(record.hash, md5String, 32+1); // die Prüfsumme wird übernommen
				record.len = current_read; // die Blocklänge 
				fwrite(inputFileBuffer+bytesRead, current_read, 1, storageFile); // Daten an Dump anfügen
				memcpy(journalMapCurrentEnd, &record, sizeof(journalentry)); // Eintrag im Journal vornehmen 
				journalMapCurrentEnd += sizeof(journalentry); // neues Journal-Ende 
				metaFileChanged = TRUE;
				if(journalEntries*sizeof(journalentry) >= journalMapLen) {
				// die Journal-Datei muss vergrößert und erneut gemappt werden 
					munmap(journalMapAdd, journalMapLen); // synchronisiert mit Dateisystem 
					journalMapAdd = mapFile(fileno(journalFile),journalMapLen, auxSpace, &journalMapLen); // remap 
					journalMapCurrentEnd = journalMapAdd + journalEntries*sizeof(journalentry);
				}
				//printf("%li;%32s;%i\n",record.block, md5String, record.len);
			} else { // DER HASH IST BEREITS BEKANNT
	printf("."); fflush(stdout);
				infoForMetaFile = hashInJournalPos; // die zeile des journals, in der der hash gefunden wurde, wird ins metafile übernommen 
			}
			// Informationen ins Metafile schreiben
			fprintf(metaFile, "%ld\n", infoForMetaFile);
			if(metaFileChanged) {
				newBytes += current_read;
				storageFileLen += current_read;
			}
			bytesRead += current_read;
		}
		if(inputFileBuffer) free(inputFileBuffer);
		if(md5String) 		free(md5String);
	}
	// Nachbereitung
	munmap(journalMapAdd, journalMapLen);
	/* Datei wieder verkleinern */
	if(ftruncate(fileno(journalFile),journalEntries*sizeof(journalentry))==-1) {
		perror("ftruncate()");
		exit(1);
	}
	laufZeit = difftime(time(NULL),startZeit);
	if(laufZeit<0.01) laufZeit=0.01;
	double speed = inputFileLenMB/laufZeit;
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
