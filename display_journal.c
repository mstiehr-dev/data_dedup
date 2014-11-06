// aus journal lesen 



#include "data_dedup.h"





int main() {
	FILE *journal = fopen(JOURNALFILE, "r+b");
	if(journal==NULL) {
		perror("could not open journalfile");
		exit(1);
	}
	long zeile = 0;
	fseek(journal,0,SEEK_END);
	long journalLength = ftell(journal);
	long records = journalLength / JOURNALLINELENGTH;
	fseek(journal,0,SEEK_SET);
	
	/* werte rausholen */ 
	while(zeile<records) {
		struct datensatz tupel;
		/* Initialisierung zum Test
		tupel.blocknummer = 1337L;
		strncpy(tupel.hash,"beliebiger hash",16);
		tupel.length = 333;
		*/
		fread(&tupel,sizeof(struct datensatz),1,journal);
		printf("%ld;%s;%d\n",tupel.blocknummer, tupel.hash, tupel.length);
		zeile++;
	}
	
	fcloseall();
	return 0;
}
