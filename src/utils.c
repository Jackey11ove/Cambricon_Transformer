#ifndef _GNU_SOURCE
  #define _GNU_SOURCE
#endif

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _USE_MATH_DEFINES
  #define _USE_MATH_DEFINES
#endif

#include <math.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/stat.h>


#ifndef USE_CMAKE_LIBS
#pragma warning(disable: 4996)
#endif

void *xmalloc(size_t size) {
  void *ptr=malloc(size);
  if(!ptr) {
    malloc_error();
  }
  return ptr;
}

void *xcalloc(size_t nmemb, size_t size) {
  void *ptr=calloc(nmemb,size);
  if(!ptr) {
    calloc_error();
  }
  return ptr;
}

void *xrealloc(void *ptr, size_t size) {
  ptr=realloc(ptr,size);
  if(!ptr) {
    realloc_error();
  }
  return ptr;
}

void del_arg(int argc, char **argv, int index) {
  int i;
  for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
  argv[i] = 0;
}

int find_arg(int argc, char* argv[], char *arg) {
  int i;
  for(i = 0; i < argc; ++i) {
    if(!argv[i]) continue;
    if(0==strcmp(argv[i], arg)) {
      del_arg(argc, argv, i);
      return 1;
    }
  }
  return 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def) {
  int i;
  for(i = 0; i < argc-1; ++i){
    if(!argv[i]) continue;
    if(0==strcmp(argv[i], arg)){
      def = atoi(argv[i+1]);
      del_arg(argc, argv, i);
      del_arg(argc, argv, i);
      break;
    }
  }
  return def;
}

float find_float_arg(int argc, char **argv, char *arg, float def) {
  int i;
  for(i = 0; i < argc-1; ++i){
    if(!argv[i]) continue;
    if(0==strcmp(argv[i], arg)){
      def = atof(argv[i+1]);
      del_arg(argc, argv, i);
      del_arg(argc, argv, i);
      break;
    }
  }
  return def;
}

char *find_char_arg(int argc, char **argv, char *arg, char *def) {
  int i;
  for(i = 0; i < argc-1; ++i){
    if(!argv[i]) continue;
    if(0==strcmp(argv[i], arg)){
      def = argv[i+1];
      del_arg(argc, argv, i);
      del_arg(argc, argv, i);
      break;
    }
  }
  return def;
}


char *basecfg(char *cfgfile) {
  char *c = cfgfile;
  char *next;
  while((next = strchr(c, '/'))) {
    c = next+1;
  }
  if(!next) while ((next = strchr(c, '\\'))) { c = next + 1; }
  c = copy_string(c);
  next = strchr(c, '.');
  if (next) *next = 0;
  return c;
}

int alphanum_to_int(char c) {
  return (c < 58) ? c - 48 : c-87;
}
char int_to_alphanum(int i) {
  if (i == 36) return '.';
  return (i < 10) ? i + 48 : i + 87;
}

void find_replace(const char* str, char* orig, char* rep, char* output) {
  char* buffer = (char*)calloc(8192, sizeof(char));
  char *p;

  sprintf(buffer, "%s", str);
  if (!(p = strstr(buffer, orig))) {  // Is 'orig' even in 'str'?
    sprintf(output, "%s", buffer);
    free(buffer);
    return;
  }

  *p = '\0';

  sprintf(output, "%s%s%s", buffer, rep, p + strlen(orig));
  free(buffer);
}

void trim(char *str) {
  char* buffer = (char*)xcalloc(8192, sizeof(char));
  sprintf(buffer, "%s", str);

  char *p = buffer;
  while (*p == ' ' || *p == '\t') ++p;

  char *end = p + strlen(p) - 1;
  while (*end == ' ' || *end == '\t') {
    *end = '\0';
    --end;
  }
  sprintf(str, "%s", p);

  free(buffer);
}

void error(const char *s) {
  perror(s);
  assert(0);
  exit(EXIT_FAILURE);
}

void malloc_error() {
  fprintf(stderr, "xMalloc error\n");
  exit(EXIT_FAILURE);
}

void calloc_error() {
  fprintf(stderr, "Calloc error\n");
  exit(EXIT_FAILURE);
}

void realloc_error() {
  fprintf(stderr, "Realloc error\n");
  exit(EXIT_FAILURE);
}

void file_error(char *s) {
  fprintf(stderr, "Couldn't open file: %s\n", s);
  exit(EXIT_FAILURE);
}

list *split_str(char *s, char delim) {
  size_t i;
  size_t len = strlen(s);
  list *l = make_list();
  list_insert(l, s);
  for(i = 0; i < len; ++i){
    if(s[i] == delim){
      s[i] = '\0';
      list_insert(l, &(s[i+1]));
    }
  }
  return l;
}

void strip(char *s) {
  size_t i;
  size_t len = strlen(s);
  size_t offset = 0;
  for(i = 0; i < len; ++i){
    char c = s[i];
    if(c==' '||c=='\t'||c=='\n'||c =='\r'||c==0x0d||c==0x0a) ++offset;
    else s[i-offset] = c;
  }
  s[len-offset] = '\0';
}


void strip_args(char *s) {
  size_t i;
  size_t len = strlen(s);
  size_t offset = 0;
  for (i = 0; i < len; ++i) {
    char c = s[i];
    if (c == '\t' || c == '\n' || c == '\r' || c == 0x0d || c == 0x0a) ++offset;
    else s[i - offset] = c;
  }
  s[len - offset] = '\0';
}

void strip_char(char *s, char bad) {
  size_t i;
  size_t len = strlen(s);
  size_t offset = 0;
  for(i = 0; i < len; ++i){
    char c = s[i];
    if(c==bad) ++offset;
    else s[i-offset] = c;
  }
  s[len-offset] = '\0';
}

void free_ptrs(void **ptrs, int n) {
  int i;
  for(i = 0; i < n; ++i) free(ptrs[i]);
  free(ptrs);
}

char *fgetl(FILE *fp) {
  if(feof(fp)) return 0;
  size_t size = 512;
  char* line = (char*)xmalloc(size * sizeof(char));
  if(!fgets(line, size, fp)){
    free(line);
    return 0;
  }

  size_t curr = strlen(line);

  while((line[curr-1] != '\n') && !feof(fp)){
    if(curr == size-1){
      size *= 2;
      line = (char*)xrealloc(line, size * sizeof(char));
    }
    size_t readsize = size-curr;
    if(readsize > INT_MAX) readsize = INT_MAX-1;
    fgets(&line[curr], readsize, fp);
    curr = strlen(line);
  }
  if(curr >= 2)
    if(line[curr-2] == 0x0d) line[curr-2] = 0x00;

  if(curr >= 1)
    if(line[curr-1] == 0x0a) line[curr-1] = 0x00;

  return line;
}

char *copy_string(char *s) {
  if(!s) {
    return NULL;
  }
  char* copy = (char*)xmalloc(strlen(s) + 1);
  strncpy(copy, s, strlen(s)+1);
  return copy;
}

list *parse_csv_line(char *line) {
  list *l = make_list();
  char *c, *p;
  int in = 0;
  for(c = line, p = line; *c != '\0'; ++c){
    if(*c == '"') in = !in;
    else if(*c == ',' && !in){
      *c = '\0';
      list_insert(l, copy_string(p));
      p = c+1;
    }
  }
  list_insert(l, copy_string(p));
  return l;
}

int count_fields(char *line) {
    int count = 0;
    int done = 0;
    char *c;
    for(c = line; !done; ++c){
        done = (*c == '\0');
        if(*c == ',' || done) ++count;
    }
    return count;
}

float *parse_fields(char *line, int n) {
  float* field = (float*)xcalloc(n, sizeof(float));
  char *c, *p, *end;
  int count = 0;
  int done = 0;
  for(c = line, p = line; !done; ++c){
      done = (*c == '\0');
      if(*c == ',' || done){
          *c = '\0';
          field[count] = strtod(p, &end);
          if(p == c) field[count] = nan("");
          if(end != c && (end != c-1 || *end != '\r')) field[count] = nan(""); //DOS file formats!
          p = c+1;
          ++count;
      }
  }
  return field;
}
