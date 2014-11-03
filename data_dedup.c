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

long countLines(FILE *f) {
	long blocks = 0;
	char line[255];
	rewind(f);
	while(fgets(line,255,f)!=NULL)
		blocks++;
	return blocks;
}

long findHashInJournal(char *hash, FILE *j) {
	/* Rückgabewert: Zeilennummer, in der der Hash gefunden wurde, also auch die Blocknummer im dumpfile
	 * sonst -1 */
	char line[255]; // hier kommen die gelesenen Zeilen rein
	long zeilenNummer = 0;
	rewind(j);
	while(fgets(line, 255, j)!=NULL) {
		if(strstr(line,hash)!=NULL) {
			// Hash gefunden 
			return zeilenNummer;
		}
		zeilenNummer++;
	}
	return -1;
}
