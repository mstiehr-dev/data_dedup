/* data_dedup.c */

#include "data_dedup.h"

char * buildString3s(const char *s1, const char *s2, const char *s3) {
	size_t l1 = strlen(s1);
	size_t l2 = strlen(s2);
	size_t l3 = strlen(s3);
	char *newStr = (char *) malloc(sizeof(char)*(l1+l2+l3+1));
	strncpy(newStr, s1, l1);
	strncpy(newStr+l1,s2,l2);
	strncpy(newStr+l1+l2,s3,l3);
	return newStr;
}

long isHashInJournal(char *hash, FILE *journal) {
	/* Rückgabewert: Zeilennummer, in der der Hash gefunden wurde, also auch die Blocknummer im dumpfile
	 * sonst -1 */
	struct datensatz tupel;
	unsigned long datensatz = 0;
	fseek(journal,0,SEEK_SET);
	while(fread(&tupel,sizeof(struct datensatz),1,journal)) {
		if(strstr(tupel.hash,hash)!=NULL) {
			// Hash gefunden 
			return datensatz;
		}
		datensatz++;
	}
	return -1;
}

long isHashInMappedJournal(char *hash, void * add, long records) {
	/* Rückgabewert: Zeilennummer, in der der Hash gefunden wurde, also auch die Blocknummer im dumpfile
	 * sonst -1 */
	struct datensatz tupel;
	unsigned long run = 0;
	while(run<records) {
		memcpy(&tupel,add+run*JOURNALLINELENGTH,JOURNALLINELENGTH);
		if(strstr(tupel.hash,hash)!=NULL) {
			// Hash gefunden 
			return run;
		}
		run++;
	}
	return -1;
}