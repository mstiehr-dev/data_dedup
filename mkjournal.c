/* mkjournal.c 
 * legt eine binäre datei an, welche 
 * 1000 unsinnige Datensätze enthält 
 */

#include "data_dedup.h"

#define size 100

int main(int argc, char **argv) {
	struct datensatz tupels[size];
	int i;
	for(i=0; i<size; i++) {
		tupels[i].blocknummer = i * 34;
		sprintf(tupels[i].hash, "hash%i", i);
		tupels[i].length = i;
	}
	char * fname = "./binaryTestJournal.dat";
	FILE *f = fopen(fname, "w+b");
	if(f==NULL) {
		fprintf(stderr, "ERROR: could not open %s", fname);
		exit(1);
	}
	fseek(f,0,SEEK_END);
	long oldFileLength = ftell(f);
	fseek(f,0,SEEK_SET);
	long newFileLength = oldFileLength + size*sizeof(struct datensatz);
	// datei etwas größer machen 
	ftruncate(fileno(f), newFileLength);
	void * add = mmap(0, newFileLength, PROT_WRITE, MAP_PRIVATE, fileno(f), 0);
	if(add==MAP_FAILED) {
		fprintf(stderr, "ERROR: could not map file\n");
		perror("mmap():");
		exit(1);
	}
	printf("mapped file, size: %i\n",size*sizeof(struct datensatz));
	for(i=0; i<size; i++) {
		memcpy(add+i*sizeof(struct datensatz), &tupels[i],sizeof(struct datensatz));
		//fwrite(&tupels[i], sizeof(struct datensatz),1,f);
		printf("%i: %ld;%s;%i\n",i, &tupels[i].blocknummer, &tupels[i].hash, &tupels[i].length);
	}
	msync(add, newFileLength, MS_SYNC);
	munmap(add, size*sizeof(struct datensatz));

// datei wieder kleiner machen 

	fclose(f);
	printf("kkthxbai\n");
	return 0;
}
