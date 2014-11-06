
/* compile via:
 * gcc versuch1.c -o v1 -lssl -lcrypto
 */


#include "data_dedup.h"

char metaChanged = FALSE; 
int main(int argc, char **argv) {
	if(argc!=2) {
		fprintf(stderr,"usage: %s <filename>\n", *argv);
		exit(1);
	}
	filename = argv[1];
	
/* Originaldatei */
	FILE *file = fopen(filename,"r");
	if(!file) {
		perror("***Fehler beim Öffnen der Datei***");
		exit(1);
	}
	/* Dateigröße ermitteln */
	fseek(file,0,SEEK_END);
	inputFileSize = ftell(file);
	fseek(file,0,SEEK_SET);
	
/* journal öffnen */
/* journal soll dann mal in den Speicher geholt werden */
	FILE *journal = fopen(JOURNALFILE, "a+b");
	if(journal==NULL) {
		perror("ERROR: could not open journal file");
		exit(1);
	}
	/* wieviele Blocks gibts schon? */
	fseek(journal,0,SEEK_END);
	journalLineNumber = (ftell(journal)+1)/JOURNALLINELENGTH; // laufende nummer der (globalen) blöcke (Zeilen im Journal)
	printf("locking file %s:\n",JOURNALFILE);
	flockfile(journal);
	
/* Storage öffnen */
	FILE *storage = fopen(STORAGEDUMP, "a+b");
	if(storage==NULL) {
		perror("ERROR: could not open storage dump file");
		exit(1);
	}
	printf("locking file %s\n",STORAGEDUMP);
	flockfile(storage);
	fseek(storage,0,SEEK_END); // Positionszeiger soll auch am Ende bleiben 
	storageBlockPosition = ftell(storage); // aktuelle Blockadresse im Storagedump

/* deduplizierten dateiindex erstellen */
	metafile = buildString3s(METADIR,basename(filename), ".meta");
	FILE *tmeta = fopen(metafile,"rt");
	if(tmeta!=NULL) {
		fprintf(stdout,"ERROR: file %s already exists.\nOverwrite? (y/N) >",metafile);
		char stdinBuf[2];
		fgets(stdinBuf,2,stdin);
		if('Y' != (*stdinBuf&0x59)) {
			// nicht überschreiben -> Abbruch
			fprintf(stdout,"Abbruch\n");
			fcloseall();
			if(metafile)
				free(metafile);
			if(dataBuffer)
				free(dataBuffer);
			return(1);
		}
		/* es soll überschrieben werden */
		fclose(tmeta);
	}
	FILE * meta = fopen(metafile,"w"); // nur schreiben, falls existent, löschen 
	if(meta==NULL) {
		perror("ERROR: could not open dedup-file for writing");
		exit(1);
	}
	
	printf("*** Start deduplication of %s (%ld Bytes)... ***\n",basename(filename),inputFileSize);
	
	/* Dateiinhalt in Speicher holen: 
	 * das könnte später noch in kleineren Schritten erfolgen */
	int i=0;
	dataBuffer = (char *) malloc(inputFileSize);
	while(fread(dataBuffer+i*CHUNKSIZE,CHUNKSIZE,1,file)>0){
		i++; // fread hört auf, wenn es nichts mehr zu lesen gibt 
	}
	
	/* durch den Datenbuffer gehen */ 
	char md5String [32+1]; // dort landet der Hash
	unsigned int current_read=0; // speichert, wie viele Bytes im aktuellen Durchlauf betrachtet werden
	int bytesRead;
	long run=0;
	long newBlocks=0;
	for(bytesRead=0;bytesRead<inputFileSize;) {
		if(++run%100==0) {
			if(metaChanged)
				printf("+");
			else
				printf("-");
			fflush(stdout);
		}
		metaChanged = FALSE;
		current_read = (CHUNKSIZE<=(inputFileSize-bytesRead)) ? CHUNKSIZE : (size_t)(inputFileSize-bytesRead);
		MD5_CTX md5context;
		unsigned char md[16];	
		if(MD5_Init(&md5context)==0) {
			perror("***Fehler beim Initialisieren von md5***");
			exit(1);
		}
		/* jetzt den Hash errechnen */
		MD5_Update(&md5context, dataBuffer+bytesRead, current_read); 
		MD5_Final(md,&md5context);
		int k;
		for(k=0; k<16; k++) // md5 String bauen 
			sprintf(&md5String[k*2],"%02x",(unsigned int) md[k]);
		// hash sichern 
		hashInJournal=isHashInJournal(md5String,journal);
		fseek(journal,0,SEEK_END);
		if(hashInJournal==-1) {
			// neuer hash -> anhängen 
			hashIDforMetafile = journalLineNumber; // der neue Zeilenindex wird ins Metafile geschrieben 
			struct datensatz tupel;
				tupel.blocknummer = storageBlockPosition;
				strncpy(tupel.hash,md5String,33);
				tupel.length = current_read; //(current_read<CHUNKSIZE)?current_read:0; // nicht mehr nötig, da Speicherung als Short
			fwrite(&tupel,sizeof(struct datensatz),1,journal);
			fwrite(dataBuffer+bytesRead, current_read, 1, storage); // block wegschreiben
			metaChanged = TRUE; 
		} else { // hash ist bereits bekannt, kann wieder verwendet werden 
			hashIDforMetafile = hashInJournal; // der vorhandene Zeilenindex wird gesichert 
		}
		/* datei-index aktualisieren */
		fprintf(meta, "%ld\n",hashIDforMetafile); // ehemals storageBlockPosition
		if(metaChanged==TRUE) {
			storageBlockPosition+=current_read;
			journalLineNumber++;
			newBlocks++;
		}
		bytesRead+=current_read;
	}
	funlockfile(journal); // Dateisperren aufheben
	funlockfile(storage);
	
	fcloseall(); // alle Datei-Ströme schließen
	if(metafile)
		free(metafile);
	if(dataBuffer)
		free(dataBuffer);
	printf("\n*** File deduplication of file %s finished ***\n\n",basename(filename));
	printf("stored %ld new blocks!\n",newBlocks);
	return 0;
}
