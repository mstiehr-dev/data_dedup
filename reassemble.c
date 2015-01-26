/* assemble.c */

#include "data_dedup.h"


int main(int argc, char **argv) {
	if(argc!=2) {
		fprintf(stderr, "usage: %s <filename>\n",*argv);
		exit(1);
	}
	
	
	char *metaFileName = *(argv+1);
	FILE *metaFile = fopen(metaFileName, "rb");
	if(metaFile==NULL) {
		fprintf(stderr,"ERROR: could not open %s\n",metaFileName);
		exit(1);
	}
	fseek(metaFile,0,SEEK_SET);
	struct stat metaFileStats; // Attribute des Journals ermitteln
	if(fstat(fileno(metaFile),&metaFileStats)==-1) {
		perror("fstat()");
		exit(1);
	}
	off_t metaFileLen = metaFileStats.st_size;
	off_t metaFileEntries = metaFileLen / sizeof(long);
	
	
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
	
	
	startZeit = time(NULL);
	
	
	// VORGEHEN: ZEILE AUS METAFILE LESEN, DIESEN DATENSATZ AUS JOURNAL HOLEN, ENTSPRECHENDEN DATENBLOCK EINLESEN UND AN ZIELDATEI ANHÄNGEN
	long metaFileInfo; // speichert, was aus Metafile gelesen wird 
	char dataBuffer[CHUNKSIZE]; // Zwischenspeicher für den Transport von Dump nach Zieldatei
	long block; // Zeileninhalt des Journals 
	journalentry journalEntry; // Speichert den jeweiligen Eintrag des Journals 
	long run=0;
	long readBytes=0;
	while(run<metaFileEntries) {
		if(fread(&metaFileInfo, sizeof(long), 1, metaFile)<=0) {
			perror("fread");
			exit(1);
		}
		fseek(journalFile, metaFileInfo*sizeof(journalentry),SEEK_SET);
		fread(&journalEntry, sizeof(journalentry), 1, journalFile);
		fseek(storageFile, journalEntry.block, SEEK_SET);
		fread(dataBuffer, journalEntry.len, 1, storageFile);
		fwrite(dataBuffer,journalEntry.len, 1, restoreFile);
		if(run++%500==0) {
			printf("+");
			fflush(stdout);
		}
		readBytes += journalEntry.len;
	}
	laufZeit = difftime(time(NULL),startZeit);
	if(laufZeit<0.5f) laufZeit=0.5f;
	double speed = (readBytes/(1024*1024.0)) / laufZeit;
	printf("\nsuccessfully reassembled \"%s\" [%.1f MB/s]\n\n",restoreFileName, speed);
	printf("please check for integrity by entering: \"diff %s <original file>\".\n", restoreFileName);
	fcloseall();
	if(restoreFileName) free(restoreFileName);
	return 0;
}
