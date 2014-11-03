/* assemble.c */

#include "data_dedup.h"


int main(int argc, char **argv) {
	if(argc!=2) {
		fprintf(stderr, "usage: %s <filename>\n",*argv);
		exit(1);
	}
	filename = *(argv+1);
	
	FILE *meta = fopen(filename, "rt");
	if(meta==NULL) {
		fprintf(stderr,"ERROR: could not open %s\n",meta);
		exit(1);
	}
	FILE *journal = fopen(JOURNALFILE,"rt");
	if(journal==NULL) {
		fprintf(stderr,"ERROR: could not open Journalfile %s\n",JOURNALFILE);
		exit(1);
	}
	restorefile = buildString3s(RESTOREDIR,basename(filename), "");
	FILE *output = fopen(restorefile,"wb");
	if(output==NULL) {
		fprintf(stderr,"ERROR: could not open file \'%s\' for writing\n",restorefile);
		exit(1);
	}
	FILE *storage = fopen(STORAGEDUMP,"rb");
	if(storage==NULL) {
		fprintf(stderr,"ERROR: could not open file \'%s\'\n",STORAGEDUMP);
		exit(1);
	}
	long metaLine, journalLine=0;
	dataBuffer = (char *) malloc((255+1)*sizeof(char));
	char *token;
	/* informationen aus metafile holen */
	while(fgets(dataBuffer,255,meta)) {
		metaLine = atoll(dataBuffer);
		printf("hole Zeile %ld\n",metaLine);
		//fseek(journal,0,SEEK_SET);
		while(journalLine<=metaLine) { //informationen aus journal holen 
			fgets(dataBuffer,255,journal);
			journalLine++;
		} // die benötigte Zeile liegt jetzt in dataBuffer
		token = strtok(dataBuffer,";"); 
		storageBlockPosition = atoll(token);
		token = strtok(NULL,";"); // wird nicht gebraucht 
		token = strtok(NULL,";");
		blockLength = atoll(token);
		if(blockLength==0) 
			blockLength = CHUNKSIZE;
		printf("muss bauen: Block %ld, Länge %ld\n",storageBlockPosition,blockLength); // läuft
		fseek(storage,storageBlockPosition,SEEK_SET);
		char readBuffer [CHUNKSIZE+1];
		fread(readBuffer,blockLength,1,storage);
		fwrite(readBuffer,blockLength,1,output);
	}
	fcloseall();
	if(dataBuffer!=NULL)
		free(dataBuffer);
	return 0;
}
