// aus journal lesen 



#include "data_dedup.h"





int main() {
	FILE *journalFile = fopen(JOURNALFILE, "r+b");
	if(journalFile==NULL) {
		perror("could not open journalfile");
		exit(1);
	}
	struct stat journalFileStats;
	if(fstat(fileno(journalFile),&journalFileStats)==-1) {
		perror("fstat()");
		exit(1);
	}
	long journalFileLen = journalFileStats.st_size;
	long records = journalFileLen / sizeof(journalentry);
	fseek(journalFile,0,SEEK_SET);
	
	/* werte rausholen */ 
	long zeile=0;
	while(zeile<records) {
		journalentry tupel;
		fread(&tupel,sizeof(journalentry),1,journalFile);
		printf("%ld;%32s;%d\n",tupel.block, tupel.hash, tupel.len);
		zeile++;
	}
	
	fcloseall();
	return 0;
}
