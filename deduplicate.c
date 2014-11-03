
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
	
/* journal öffnen */
	FILE *journal = fopen(JOURNALFILE, "a+t");
	if(journal==NULL) {
		perror("ERROR: could not open journal file");
		exit(1);
	}
	/* wieviele Blocks gibts schon? */
	journalLineNumber = countLines(journal); // laufende nummer der (globalen) blöcke (Zeilen im Journal)
	printf("locking file %s:\n",JOURNALFILE);
	flockfile(journal);
	
/* Storage öffnen */
	FILE *storage = fopen(STORAGEDUMP, "a+");
	if(storage==NULL) {
		perror("ERROR: could not open storage dump file");
		exit(1);
	}
	printf("locking file %s\n",STORAGEDUMP);
	flockfile(storage);
	fseek(storage,0,SEEK_END);
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
	while(fread(dataBuffer+i*CHUNKSIZE,1,CHUNKSIZE,file)>0){
		i++; // fread hört auf, wenn es nichts mehr zu lesen gibt 
	}
	
	/* durch den Datenbuffer gehen */ 
	char md5String [32+1];
	size_t current_read=0;
	int showProgress = 0;
	int bytesRead;
	long run=0;
	for(bytesRead=0;bytesRead<inputFileSize;) {
		if(++run%100==0)
			printf("+");
			fflush(stdout);
		metaChanged = FALSE;
		current_read = (CHUNKSIZE<=(inputFileSize-bytesRead)) ? CHUNKSIZE : (size_t)(inputFileSize-bytesRead);
		MD5_CTX md5context;
		unsigned char md[16];	
		if(MD5_Init(&md5context)==0) {
			perror("***Fehler beim Initialisieren von md5***");
			exit(1);
		}
		//printf("%s\n",(char *) dataBuffer+i);
		/* jetzt den Hash errechnen */
		MD5_Update(&md5context, dataBuffer+bytesRead, current_read); 
		MD5_Final(md,&md5context);
		int k;
		for(k=0; k<16; k++) // md5 String bauen 
			sprintf(&md5String[k*2],"%02x",(unsigned int) md[k]);
		//printf("[%-4i] - [%-4li] --> %s\n",i,current_read, md5String);
		// hash sichern 
		hashNummer=findHashInJournal(md5String,journal);
		if(hashNummer==-1) {
			// neuer hash -> anhängen 
			schreibNummer = journalLineNumber; // der neue Zeilenindex wird ins Metafile geschrieben 
			fseek(journal,0,SEEK_END);
			fprintf(journal,"%ld;%s;%i;\n",storageBlockPosition,md5String,(current_read<CHUNKSIZE)?current_read:0);// die journalLineNumber im Journal dient erstmal der Kontrolle, ist nachher aber überflüssig (zeilennummer = journalLineNumber)
			//printf("***new block***\n");
			fwrite(dataBuffer+bytesRead, current_read, 1, storage); // block wegschreiben
			metaChanged = TRUE;
		} else { // hash ist bereits bekannt, kann wieder verwendet werden 
			schreibNummer = hashNummer; // der vorhandene Zeilenindex wird gesichert 
			//printf("already got this one! it's stored in line %ld\n",schreibNummer);
		}
		/* datei-index aktualisieren */
		fprintf(meta, "%ld\n",schreibNummer); // ehemals storageBlockPosition
		if(metaChanged==TRUE) {
			storageBlockPosition+=current_read;
			journalLineNumber++;
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
	printf("*** File deduplication finished ***\n\n");
	return 0;
}
