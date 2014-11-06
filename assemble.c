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
		fprintf(stderr,"ERROR: could not open %s\n",filename);
		exit(1);
	}
	FILE *journal = fopen(JOURNALFILE,"rb");
	if(journal==NULL) {
		fprintf(stderr,"ERROR: could not open Journalfile %s\n",JOURNALFILE);
		exit(1);
	}
	char *restorefilename = buildString3s(RESTOREDIR,basename(filename), "");
	*(restorefilename+(strlen(restorefilename)-5)) = '\0';
	FILE *output = fopen(restorefilename,"wb");
	if(output==NULL) {
		fprintf(stderr,"ERROR: could not open file \'%s\' for writing\n",restorefilename);
		exit(1);
	}
	FILE *storage = fopen(STORAGEDUMP,"rb");
	if(storage==NULL) {
		fprintf(stderr,"ERROR: could not open file \'%s\'\n",STORAGEDUMP);
		exit(1);
	}
	long metaLine, journalLine=0;
	journalLineBuffer = (char *) malloc((255+1)*sizeof(char));
	char *token;
	/* informationen Zeilenweise aus metafile holen */
	struct datensatz tupel;
	long run=0;
	while(fgets(journalLineBuffer,255,meta)) {
		if(++run%50==0) {
			fflush(stdout);
			printf("*");
		}
		metaLine = atoll(journalLineBuffer); // dieser Datensatz wird aus dem Journal benötigt
		fseek(journal,metaLine*sizeof(struct datensatz),SEEK_SET);
		fread(&tupel,sizeof(struct datensatz),1,journal);
		fseek(storage,tupel.blocknummer,SEEK_SET);
		char readBuffer [CHUNKSIZE+1];
		fread(readBuffer,tupel.length,1,storage);
		fwrite(readBuffer,tupel.length,1,output);
	}
	printf("\nrestored file successfully: %s\n\n",restorefilename);
	fcloseall();
	if(journalLineBuffer!=NULL)
		free(journalLineBuffer);
	if(restorefilename)
		free(restorefilename);
	return 0;
}
