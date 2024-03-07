#ifndef UTILS_H
#define UTILS_H
#include "simulator.h"
#include "list.h"

#include <stdio.h>
#include <time.h>

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#ifdef __cplusplus
extern "C" {
#endif

void *xmalloc(size_t size); //带报错的malloc
void *xcalloc(size_t nmemb, size_t size); //带报错的calloc，分配并初始化内存空间
void *xrealloc(void *ptr, size_t size); //带报错的ralloc，重新分配内存的size大小

// 删除参数数组中指定索引位置的参数后，将后续参数向前移动一个位置，并在最后添加一个标志参数结束的 0
void del_arg(int argc, char **argv, int index);
// 在参数数组中查找指定的参数，如果找到则删除该参数并返回1，如果未找到则返回0
int find_arg(int argc, char* argv[], char *arg);
/* 在参数数组中查找指定的参数，如果找到就将该参数后面的一个参数变为 int/float/char* 类型返回，
   并删除该参数和后一个参数，没找到就返回原来的def */
int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
char *find_char_arg(int argc, char **argv, char *arg, char *def);

// 路径中提取文件名：从 "/path/to/config_file.cfg" 中提取出文件名 config_file，并去掉扩展名
char *basecfg(char *cfgfile);
// 将数字和字母类型的char转为数字类型，0-9，a=11，b=12...
int alphanum_to_int(char c);
char int_to_alphanum(int i);

// 在给定的字符串 str 中查找子字符串 orig，并将其替换为字符串 rep。替换后的结果将存储在 output 指向的内存中。
LIB_API void find_replace(const char* str, char* orig, char* rep, char* output);
// 去除字符串 str 前后的空格和制表符
void trim(char *str);

int read_int(int fd);
void write_int(int fd, int n);
void read_all(int fd, char *buffer, size_t bytes);
void write_all(int fd, char *buffer, size_t bytes);

// 各种错误信息报错
void error(const char *s);
void malloc_error();
void calloc_error();
void realloc_error();
void file_error(char *s);

// 将一个字符串存在新建的链表头，并按照分隔符delim的分割，每次遇到分隔符都把该分隔符后面的字符串插入到链表中
list *split_str(char *s, char delim);
// 去除字符串 s 中的空格、制表符、换行符和回车符，并将结果存储在原始的字符串 s 中
void strip(char *s);
// 跟strip作用一致
void strip_args(char *s);
// 去除参数字符串 s 中出现的指定字符 bad，并将结果存储在原始的字符串 s 中
void strip_char(char *s, char bad);
// 释放一个指针数组 ptrs 中的所有内存块，并最终释放这个指针数组本身所占用的内存，n为ptrs的size
LIB_API void free_ptrs(void **ptrs, int n);

// 从文件流 fp 中读取一行文本(读到'\n'为止)，并返回一个动态分配的字符数组，存储该行文本的内容
char *fgetl(FILE *fp);
// 复制输入的字符串 s 并返回一个新的动态分配的字符串
char *copy_string(char *s);
// 没用上
list *parse_csv_line(char *line);
int count_fields(char *line);
float *parse_fields(char *line, int n);

#ifdef __cplusplus
}
#endif

#endif
