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
	journalentry *tupel = (journalentry *) add; // zeigt nun auf den ersten Datensatz
	unsigned long line = 0;
	while(line<records) {
		/*
		memcpy(&tupel,tempAdd,sizeof(journalentry)); 
		if(strstr(tupel.hash,hash)!=NULL) {
			// Hash gefunden
			return line;
		}
		line++;
		tempAdd += sizeof(journalentry); */
		// besser: 
		if(memcmp4l(tupel->hash, hash)==0) {
			// TREFFER!
			return line;
		} /* else */
		line++;
		tupel++;
	}
	return -1;
}

int memcmp4l(char *s, char *t) { // gibt 1 zurück bei Unterscheidung
	int i = 32/sizeof(long); // 4
	long *l1 = (long*)s;
	long *l2 = (long*)t;
	while(i--) {
		if(*l1!=l2)
			return 1;
		l1++;
		l2++;
	}
	return 0;
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
		perror("malloc() failed in getRandString()");
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


