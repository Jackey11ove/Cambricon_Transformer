#ifndef OPTION_H
#define OPTION_H
#include "simulator.h"
#include "list.h"

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

#ifdef __cplusplus
extern "C" {
#endif

// 好像没用上
list *read_data_cfg(char *filename);
// 将str s以等号为界限分出key和val(s: key=val)，将该键值对作为一个kvp插入list option中，若未找到=返回0，否则返回1
int read_option(char *s, list *options);
// 根据key和val参数初始化kvp并将其作为一个node的*val插入list中
void option_insert(list *l, char *key, char *val);
// 在节点node->val都是kvp的链表l中查找第一个键为key的kvp，将该kvp的use值设为1，并返回该kvp的kvp->val
char *option_find(list *l, char *key);
// 好像没用上
char *option_find_str(list *l, char *key, char *def);
char *option_find_str_quiet(list *l, char *key, char *def);
// 链表中寻找对应key的val值，找到了就将其转换为整型返回，未找到就返回def值
int option_find_int(list *l, char *key, int def);
// 链表中寻找对应key的val值，找到了就将其转换为整型返回，未找到就返回def值，但不在输出流上显示未找到情况下的日志
int option_find_int_quiet(list *l, char *key, int def);
// 链表中寻找对应key的val值，找到了就将其转换为float返回，未找到就返回def值
float option_find_float(list *l, char *key, float def);
float option_find_float_quiet(list *l, char *key, float def);
// 在list l中寻找unused的节点，打印相关键值对
void option_unused(list *l);

#ifdef __cplusplus
}
#endif
#endif
