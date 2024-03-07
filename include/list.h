#ifndef LIST_H
#define LIST_H

/*这两个结构体的用法为：node结构体形成链表，而list则将front指向链表头，back指向链表尾*/
typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;


#ifdef __cplusplus
extern "C" {
#endif

// 创建并初始化list结构 
list *make_list();

int list_find(list *l, void *val);

// 在链表中插入节点，更新list->back指针；如果链表为空则将front、back都指向该新节点
void list_insert(list *, void *);
// 将链表的各个节点值变成一个array存下来，由于节点值为字符串，array为二维数组
void **list_to_array(list *l);

// 将链表中各个节点的val指针free掉
void free_list_val(list *l);
// 将list和list下的各个节点都free掉
void free_list(list *l);
// 感觉跟free_list_val没啥区别，都是把链表的val指针free
void free_list_contents(list *l);
// 好像没啥用
void free_list_contents_kvp(list *l);

#ifdef __cplusplus
}
#endif
#endif
