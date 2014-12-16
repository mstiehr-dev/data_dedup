/* findDataOnHost.c */
/* gcc -lrt */


#include "data_dedup.h" 

#ifndef HAYSTACK
  #define HAYSTACK 1000
#endif



int main(int argc, char **argv) {
	srand(time(NULL));
	#ifndef treffer
	unsigned int treffer = randFloat() * HAYSTACK; // Dieser Datensatz wird nachher im HAYSTACK gesucht
	#endif
	// --- lokalen Speicher bereitstellen und initialisieren 
	journalentry * host_data = (journalentry *) malloc(HAYSTACK*sizeof(journalentry));
	memset(host_data, 0, HAYSTACK*sizeof(journalentry));
	int i;
	for(i=0; i<HAYSTACK; i++) {
		(host_data+i)->block = LONG_MAX * randFloat();
		(host_data+i)->len = SHRT_MAX * randFloat();
		char *tString = randString(32);
		strncpy((host_data+i)->hash,tString,32);
		free(tString);
	}
	
	journalentry findMe; 
	memcpy(&findMe, host_data+treffer, sizeof(journalentry));
	
	// hash finden 
	static struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
	for(i=0; i<HAYSTACK; i++) {
		if(memcmp(findMe.hash, (host_data+i)->hash, 32)==0) {
			printf("got it in segment #%d!\n",i);
			break;
		}
	}
	static struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    long nsecs = end.tv_nsec - start.tv_nsec;
    long usecs = nsecs/1000;
    long msecs = usecs/1000;
	printf("it took %ldms to find this hash within a haystack of %d hashes!\n",msecs, (int)HAYSTACK);
	// aufräumen 
	free(host_data);
	
	printf("kkthxbai\n");
	return 0;
}



char randChar() {
	//liefert ein zufälliges druckbares Zeichen 
	/* Druckbare Zeichen beginnen bei 32 (A) und enden bei 126 (~) -> Spanne von 94 */
	const char start = 'A';
	const char end   = '~';
	const char range = end-start;
	char  c = 32 + randFloat()*range; 
	return c;
}
char * randString(size_t n) {
	// liefert eine zufällige Zeichenkette 
	char *str = (char *)malloc(n*sizeof(char)+1);
	if(str==NULL) {
		perror("malloc() failed in randString()");
		exit(1);
	}
	str[n] = '\0'; // Stringende 
	while(n--)
		str[n] = randChar();
	return str;
}

float randFloat() { // liefert eine Zufallszahl zwischen 0 und 1 (inklusive) 
	return ((float)rand())/RAND_MAX;
}
