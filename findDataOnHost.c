/* findDataOnHost.c */
/* gcc findDataOnHost.c -lrt -o onHost -DHAY_STACK=10000000 -DTreffer=HAY_STACK-1*/
/* ./onHost <haystack> <hit> */
/* for debug mode: -DDEBUG   */
#include "data_dedup.h" 
#include <unistd.h>

int main(int argc, char **argv) {

	unsigned int runs;
	unsigned int haystack = 10000000;
	unsigned int treffer;//  = haystack -1;
	opterr = 0;
	char c;
	while((c=getopt(argc, argv, "c:?h:"))!=-1) { // : -> argument required
		switch(c) {
			case 'h':	if(optarg) haystack = atoi(optarg); break;
			case 'c':	if(optarg) runs		 = atoi(optarg); break;
			case '?':	printf("gooby pls halp\n"); break;
			default:
				printf("usage: %s -h <size of _haystack> -b <_blocks> -t <_threads per block>\n",argv[0]);
				exit(1);
		}
	}
	srand(time(NULL));
	if(argc==2)
		haystack = atoi(*(argv+1));
	treffer = getRandFloat() * haystack;
	// --- lokalen Speicher bereitstellen und initialisieren 
	journalentry * host_data = (journalentry *) malloc(haystack*sizeof(journalentry));
	memset(host_data, 0, haystack*sizeof(journalentry));
	int i;
	static struct timespec start;
    #ifdef DEBUG 
    printf("Start Initialisierung: %d:%ld\n",start.tv_sec, start.tv_nsec);
    clock_gettime(CLOCK_MONOTONIC, &start);
    #endif
    /* Initialisierung */
	for(i=0; i<haystack; i++) {
		(host_data+i)->block = LONG_MAX * getRandFloat();
		(host_data+i)->len = SHRT_MAX * getRandFloat();
		char *tString = getRandString(32);
		strncpy((host_data+i)->hash,tString,32);
		free(tString);
	}
	journalentry findMe; 
	memcpy(&findMe, host_data+treffer, sizeof(journalentry));
	
	/* Suche */
    clock_gettime(CLOCK_MONOTONIC, &start);
    #ifdef DEBUG 
    printf("Start Suche: %d:%09ld\n",start.tv_sec, start.tv_nsec);
	#endif
	while(runs--) {
		for(i=0; i<haystack; i++) {
			if(mymemcmp(findMe.hash, (host_data+i)->hash, 32)==0) {
				//printf("found it in segment #%d!\n",i);
				break;
			}
		}
	}
	
	static struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    #ifdef DEBUG
    printf("Ende: %d:%ld\n",end.tv_sec, end.tv_nsec);    
    #endif
    long nsecs = end.tv_nsec - start.tv_nsec;
    long usecs = nsecs/1000;
    long msecs = usecs/1000;
    int sec = end.tv_sec - start.tv_sec;
    if(msecs<0) {
    	msecs *= -1;
    	sec--;
    }
	printf("it took %d.%.6lds to find this hash within a haystack of %u hashes!\n",sec,msecs, haystack);
	// aufräumen 
	free(host_data);
	return 0;
}

int mymemcmp(const void *s1, const void *s2, size_t len) {
	const long *l1 = (long*)s1, *l2 = (long*)s2;
	len = (len+sizeof(long)-1)/sizeof(long);
	while(len--) {
		if(*l1 != *l2) 
			return(-1);
		l1++;
		l2++;
	}
	return 0;
}

char getRandChar() {
	//liefert ein zufälliges druckbares Zeichen 
	/* Druckbare Zeichen beginnen bei 32 (A) und enden bei 126 (~) -> Spanne von 94 */
	const char start = 'A';
	const char end   = '~';
	const char range = end-start;
	char  c = 32 + getRandFloat()*range; 
	return c;
}
char * getRandString(size_t n) {
	// liefert eine zufällige Zeichenkette 
	char *str = (char *)malloc(n*sizeof(char)+1);
	if(str==NULL) {
		perror("malloc() failed in randString()");
		exit(1);
	}
	str[n] = '\0'; // Stringende 
	while(n--)
		str[n] = getRandChar();
	return str;
}

float getRandFloat() { // liefert eine Zufallszahl zwischen 0 und 1 (inklusive) 
	return ((float)rand())/RAND_MAX;
}
