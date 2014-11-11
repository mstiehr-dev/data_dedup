


/* memcpy.c 
 * Testen: eine Datei in den Speicher holen und ausgeben
 */

#define _GNU_SOURCE

#include "data_dedup.h"

int main(int argc, char **argv) {
	if(argc!=2) {
		fprintf(stderr,"usage: %s <filename>\n",*argv);
		exit(1);
	}
	char *fname = *(argv+1);
	FILE *f = fopen(fname, "rb");
	fseek(f,0,SEEK_END);
	long flength = ftell(f);
	fseek(f,0,SEEK_SET);

	void * pa = mmap(0,flength,PROT_READ,MAP_PRIVATE,fileno(f),0);
	fclose(f);
	if(pa==MAP_FAILED) {
		perror("ERROR: could not map your file!");
		exit(1);
	}
	struct datensatz tupel;
	tupel.blocknummer = 0L;
	strncpy(tupel.hash,"default hash",13);
	tupel.length = 0;
	int run=0;
	while(run < (flength/sizeof(struct datensatz))) {
		memcpy(&tupel,pa+run*sizeof(struct datensatz),sizeof(struct datensatz));
		printf("Datensatz: %ld;%s;%i\n",tupel.blocknummer, tupel.hash, tupel.length);
		run++;
	}

	munmap(pa,flength);
	return 0;
}
