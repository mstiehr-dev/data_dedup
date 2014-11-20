/* deduplicate.c */


#include "data_dedup.h"



int main(int argc, char **argv) {
	if(argc!=2) {
		fprintf(stderr, "usage: %s <filename>\n",*argv);
		exit(1);
	}
	char *inputFileName = *(argv+1);
	
	
	// DIE ZU DEDUPLIZIERENDE DATEI
	FILE *inputFile = fopen(inputFileName, "rb");
	if(inputFile==NULL) {
		fprintf(stderr,"ERROR: could not open %s!\n",inputFileName);
		perror("fopen()");
		exit(1);
	}
	struct stat inputFileStats;
	if(fstat(fileno(inputFile),&inputFileStats)==-1) {
		perror("fstat()");
		exit(1);
	}
	off_t inputFileLen = inputFileStats.st_size;
	double inputFileLenMB = inputFileLen/(1024*1024.0);
	
	
	// DIE INDEXDATEI ÜBER ALLE HASHES (JOURNAL)
	FILE *journalFile = fopen(JOURNALFILE,"a+b");
	if(journalFile==NULL) {
		fprintf(stderr,"ERROR: could not open %s!\n",JOURNALFILE);
		perror("fopen()");
		exit(1);
	}
	struct stat journalFileStats;
	if(fstat(fileno(journalFile),&journalFileStats)==-1) {
		perror("fstat()");
		exit(1);
	}
	off_t journalFileLen = journalFileStats.st_size;
	
	
	// STELLVERTRETER FÜR DIE DEDUPLIZIERTE DATEI (METAFILE) 
	char *metaFileName = (char *)buildString3s(METADIR,basename(inputFileName), ".meta");
	FILE *metaFile = fopen(metaFileName,"rt");
	if(metaFile!=NULL) { // Datei existiert bereits 
		fprintf(stdout,"WARNING: file \'%s\' already exists.\nOverwrite? (y/N) >",metaFileName);
		char stdinBuf[2];
		fgets(stdinBuf,2,stdin);
		if('Y' != (*stdinBuf&0x59)) {// nicht überschreiben -> Abbruch
			fprintf(stdout,"=> Abbruch\n");
			fcloseall();
			if(metaFileName) free(metaFileName);
			return(1);
		}
		/* es soll überschrieben werden */
		fclose(metaFile);
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
	fseek(storageFile,0,SEEK_END);
	struct stat storageFileStats;
	if(fstat(fileno(storageFile),&storageFileStats)==-1) {
		perror("fstat()");
		exit(1);
	}
	off_t storageFileLen = storageFileStats.st_size;
	
	// DAS JOURNAL MAPPEN (zusätzlicher Platz für 100 Einträge)
	off_t journalMapLen;
	void *journalMapAdd = mapFile(fileno(journalFile),journalFileLen, auxSpace, &journalMapLen);
	void *journalMapCurrentAdd = journalMapAdd + journalFileLen; // Hilfszeiger soll ans Dateiende zeigen 
	// STATISTIK
	off_t journalEntries = journalFileLen / sizeof(journalentry);	


// BEGINN DER VERARBEITUNG
startZeit = time(NULL);	
long newBytes = 0; // Blöcke, die neu ins Journal aufgenommen wurden 
char * inputFileBuffer;
	// DIE EINGABEDATEI EINLESEN
	unsigned int bytesBufferSize = 1*1024*1024; // 1 MB
	off_t bytesActuallyBuffered = 0L;
	off_t bytesBuffered;
	for(bytesBuffered = 0L; bytesBuffered<inputFileLen; bytesBuffered+=bytesActuallyBuffered) {
		inputFileBuffer = malloc(sizeof(char)*bytesBufferSize);
		if(inputFileBuffer==NULL) {
			perror("ERROR: could not allocate memory");
			exit(1);
		}
		fread(inputFileBuffer,1,bytesBufferSize,inputFile);
		bytesActuallyBuffered = (inputFileLen-bytesBuffered >= bytesBufferSize) ? bytesBufferSize : (inputFileLen-bytesBuffered);
	// FÜR JEDEN CHUNK EINEN HASH BERECHNEN UND MIT DEM JOURNAL VERGLEICHEN 
		char *md5String = malloc(sizeof(char)*(32+1));
		if(md5String==NULL) {
			perror("ERROR: could not allocate memory for md5String");
			exit(1);
		}
		int current_read = 0; // wie viele Bytes aktuell vorhanden sind 
		size_t bytesRead; // Summe der konsumierten Bytes 
		char metaFileChanged = FALSE;
		long infoForMetaFile; // enthält die jeweilige Zeilennummer des Journals
		MD5_CTX md5Context;   // Struktur für die Hash-Berechnung
		unsigned char md[16];
		printf("deduplicating \"%s\" [%.3f MB]\n",inputFileName, inputFileLenMB);
		for(bytesRead=0; bytesRead<bytesActuallyBuffered;) {
			metaFileChanged = FALSE;
			current_read = (CHUNKSIZE<=(bytesActuallyBuffered-bytesRead)) ? CHUNKSIZE : bytesActuallyBuffered-bytesRead;
			if(MD5_Init(&md5Context)==0) {
				perror("MD5_Init()");
				exit(1);
			}
			MD5_Update(&md5Context, inputFileBuffer+bytesRead, current_read);
			MD5_Final(md, &md5Context);
			int i;
			for(i=0;i<16;i++)  // String bauen 
				sprintf(md5String+2*i, "%02x", (unsigned int) md[i]);	
			// Testen, ob der errechnete Hash bereits bekannt ist
			long hashInJournalPos = isHashInMappedJournal(md5String, journalMapAdd, journalEntries);
			if(hashInJournalPos==-1) { // DER HASH IST UNBEKANNT -> MUSS ANGEFÜGT WERDEN 
	printf("+");
				infoForMetaFile = journalEntries; // in diesem Datensatz wird sich der neue Hash befinden
				journalentry record; // neuen Eintrag bauen 
				record.block = storageFileLen;
				strncpy(record.hash, md5String, 32+1);
				record.len = current_read;
				fwrite(inputFileBuffer+bytesRead, current_read, 1, storageFile); // Daten an Dump anfügen
				memcpy(journalMapCurrentAdd, &record, sizeof(journalentry)); // Eintrag im Journal
				journalMapCurrentAdd += sizeof(journalentry);
				metaFileChanged = TRUE;
				journalEntries++;
				if(journalEntries*sizeof(journalentry) >= journalMapLen) {
				// die Journal-Datei muss vergrößert und erneut gemappt werden 
					munmap(journalMapAdd, journalMapLen);
					journalMapAdd = mapFile(fileno(journalFile),journalMapLen, auxSpace, &journalMapLen);
					journalMapCurrentAdd = journalMapAdd + journalEntries*sizeof(journalentry);
				}
				//printf("%li;%32s;%i\n",record.block, md5String, record.len);
			} else { // DER HASH IST BEREITS BEKANNT
	printf(".");
				infoForMetaFile = hashInJournalPos;
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
	if(laufZeit==0) laufZeit=1.0;
	double speed = inputFileLenMB/laufZeit;
	if(metaFileName) free(metaFileName);
	fcloseall();
	printf("\n\n*** successfully deduplicated \"%s\" in %.1fs [%.3f MB/s] ***\n", inputFileName, laufZeit, speed);
	printf("*** added %ld Bytes to storage dump ***\n",newBytes);
	return 0;
}
