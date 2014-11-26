/* assemble.c */

#include "data_dedup.h"


int main(int argc, char **argv) {
	if(argc!=2) {
		fprintf(stderr, "usage: %s <filename>\n",*argv);
		exit(1);
	}
	
	
	char *metaFileName = *(argv+1);
	FILE *metaFile = fopen(metaFileName, "rt");
	if(metaFile==NULL) {
		fprintf(stderr,"ERROR: could not open %s\n",metaFileName);
		exit(1);
	}
	fseek(metaFile,0,SEEK_SET);
	
	
	FILE *journalFile = fopen(JOURNALFILE,"rb");
	if(journalFile==NULL) {
		fprintf(stderr,"ERROR: could not open Journalfile %s\n",JOURNALFILE);
		exit(1);
	}
	fseek(journalFile,0,SEEK_SET);
	
	
	char *restoreFileName = (char *)buildString3s(RESTOREDIR,basename(metaFileName), "");
	*(restoreFileName+(strlen(restoreFileName)-5)) = '\0'; //".meta" abschneiden
	FILE *restoreFile = fopen(restoreFileName,"wb");
	if(restoreFile==NULL) {
		fprintf(stderr,"ERROR: could not open file \'%s\' for writing\n",restoreFileName);
		exit(1);
	}
	
	
	FILE *storageFile = fopen(STORAGEDUMP,"rb");
	if(storageFile==NULL) {
		fprintf(stderr,"ERROR: could not open file \'%s\'\n",STORAGEDUMP);
		exit(1);
	}
	fseek(storageFile,0,SEEK_SET);
	printf(" * reassembling \"%s\"\n",restoreFileName);
	
	// VORGEHEN: ZEILE AUS METAFILE LESEN, DIESEN DATENSATZ AUS JOURNAL HOLEN, ENTSPRECHENDEN DATENBLOCK EINLESEN UND AN ZIELDATEI ANHÄNGEN
	char metaFileBuffer[sizeof(long)+sizeof('\n')+1]; // speichert, was aus Metafile gelesen wird 
	char dataBuffer[CHUNKSIZE]; // Zwischenspeicher für den Transport von Dump nach Zieldatei
	long block;
	journalentry journalEntry; // Speichert den jeweiligen Eintrag des Journals 
	long run=0;
	while(fgets(metaFileBuffer, sizeof(metaFileBuffer), metaFile)) {
		block = atol(metaFileBuffer);
		fseek(journalFile, block*sizeof(journalentry),SEEK_SET);
		fread(&journalEntry, sizeof(journalentry), 1, journalFile);
		fseek(storageFile, journalEntry.block, SEEK_SET);
		fread(dataBuffer, journalEntry.len, 1, storageFile);
		fwrite(dataBuffer,journalEntry.len, 1, restoreFile);
		if(run++%500==0) {
			printf("+");
			fflush(stdout);
		}			
	}
	printf("\nsuccessfully reassembled \"%s\"\n\n",restoreFileName);
	fcloseall();
	if(restoreFileName) free(restoreFileName);
	return 0;
}
