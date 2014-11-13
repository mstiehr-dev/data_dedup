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
		perror("fopen()");
		exit(1);
	}
	struct stat inputFileStats;
	if(fstat(fileno(inputFile),&inputFileStats)==-1) {
		perror("fstat()");
		exit(1);
	}
	off_t inputFileLen = inputFileStats.st_size;
	
	
	// DIE INDEXDATEI ÜBER ALLE HASHES (JOURNAL)
	FILE *journalFile = fopen(JOURNALFILE,"a+b");
	if(journalFile==NULL) {
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
		perror("fopen()");
		exit(1);
	}
	
	
	// DATENHALDE ÖFFNEN
	FILE *storageFile = fopen(STORAGEDUMP, "a+b");
	if(storageFile==NULL) {
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
	
	
	// DIE EINGABEDATEI EINLESEN 
	char *inputFileBuffer = malloc(sizeof(char)*inputFileLen);
	int i=0;
	while(fread(inputFileBuffer+i*100*CHUNKSIZE, CHUNKSIZE, 100, inputFile))
		i++;
	// FÜR JEDEN CHUNK EINEN HASH BERECHNEN UND MIT DEM JOURNAL VERGLEICHEN 
	char md5String[32+1];
	int current_read = 0; // wie viele Bytes aktuell vorhanden sind 
	size_t bytesRead; // Summe der konsumierten Bytes 
	long run = 0;  // Durchlauf
	long newBlocks = 0; // Blöcke, die neu ins Journal aufgenommen wurden 
	char metaFileChanged;
	long infoForMetaFile;
	MD5_CTX md5Context;
	unsigned char md[16];
	printf("deduplicating \"%s\"\n",inputFileName);
	for(bytesRead=0; bytesRead<inputFileLen;) {
		metaFileChanged = FALSE;
		current_read = (CHUNKSIZE<=(inputFileLen-bytesRead)) ? CHUNKSIZE : inputFileLen-bytesRead;
		if(MD5_Init(&md5Context)==0) {
			perror("MD5_Init()");
			exit(1);
		}
		MD5_Update(&md5Context, inputFileBuffer+bytesRead, current_read);
		MD5_Final(md, &md5Context);
		for(i=0;i<16;i++) 
			sprintf(md5String+2*i, "%02x", (unsigned int) md[i]);	
		// Testen, ob der errechnete Hash bereits bekannt ist
		long hashInJournalPos = isHashInMappedJournal(md5String, journalMapAdd, journalEntries);
		if(hashInJournalPos==-1) { // DER HASH IST UNBEKANNT -> MUSS ANGEFÜGT WERDEN 
printf("!");
			infoForMetaFile = journalEntries; // in diesem Datensatz wird sich der neue Hash befinden
			journalentry record; // neuen Eintrag bauen 
			record.block = storageFileLen;
			strncpy(record.hash, md5String, 32+1);
			record.len = current_read;
			fwrite(inputFileBuffer, current_read, 1, storageFile); // Daten anfügen
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
			newBlocks++;
			storageFileLen += current_read;
		}
		bytesRead += current_read;
		if(run++%500==0) {
			printf("+");
			fflush(stdout);
		}	
	}
	munmap(journalMapAdd, journalMapLen);
	/* Datei wieder verkleinern */
	if(ftruncate(fileno(journalFile),journalEntries*sizeof(journalentry))==-1) {
		perror("ftruncate()");
		exit(1);
	}
	if(metaFileName) free(metaFileName);
	fcloseall();
	printf("\n\n*** successfully deduplicated \"%s\" ***\n", inputFileName);
	return 0;
}
