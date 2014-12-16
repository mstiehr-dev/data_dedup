/* data_dedup.c */

#include "data_dedup.h"

const char * buildString3s(const char *s1, const char *s2, const char *s3) {
	size_t l1 = strlen(s1);
	size_t l2 = strlen(s2);
	size_t l3 = strlen(s3);
	char *newStr = (char *) malloc(sizeof(char)*(l1+l2+l3+1));
	strncpy(newStr, s1, l1);
	strncpy(newStr+l1,s2,l2);
	strncpy(newStr+l1+l2,s3,l3);
	return newStr;
}

long isHashInMappedJournal(char *hash, void * add, long records) {
	/* Rückgabewert: Zeilennummer, in der der Hash gefunden wurde, also auch die Blocknummer im dumpfile
	 * sonst -1 */
	journalentry tupel;
	void * tempAdd = add;
	unsigned long line = 0;
	while(line<records) {
		memcpy(&tupel,tempAdd,sizeof(journalentry));
		if(strstr(tupel.hash,hash)!=NULL) {
			// Hash gefunden
			return line;
		}
		line++;
		tempAdd += sizeof(journalentry);
	}
	return -1;
}



void * mapFile(int fd, off_t len, int aux, off_t *saveLen) {
	off_t tempLen = len+aux;
	if(ftruncate(fd,tempLen)==-1) {
		perror("ftruncate()");
		exit(1);
	}
	void *add = mmap(NULL, tempLen, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if(add==MAP_FAILED) {
		perror("mmap()");
		printf("%s\n",strerror(errno));
		exit(1);
	}	
	*saveLen = tempLen;
	return add;
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
