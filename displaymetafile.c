/* displaymetafile.c */


#include "data_dedup.h"

int main(int argc, char **argv) {
	if(argc!=2) {
		fprintf(stderr, "usage: %s <metafile>\n", argv[0]);
		exit(1);
	}
	FILE *metaFile = fopen(argv[1], "rb");
	if(metaFile==NULL) {
		perror("could not open metaFile");
		exit(1);
	}
	struct stat metaFileStats;
	if(fstat(fileno(metaFile),&metaFileStats)==-1) {
		perror("fstat()");
		exit(1);
	}
	long metaFileLen = metaFileStats.st_size;
	long records = metaFileLen / sizeof(journalentry);
	fseek(metaFile,0,SEEK_SET);
	
	/* werte rausholen */ 
	long zeile=0;
	while(zeile<records) {
		long entry;
		fread(&entry,sizeof(entry),1,metaFile);
		printf("%ld\n",entry);
		zeile++;
	}
	
	fcloseall();
	return 0;
}
