OS := $(shell uname)

VPATH=./src/
EXEC=simulator
OBJDIR=./obj/

ifeq ($(OS),Darwin) #MAC
CC=clang
else
CC=gcc
endif

OPTS=-Ofast
COMMON= -Iinclude
CFLAGS=-Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -fPIC

CFLAGS+=$(OPTS)

OBJ=utils.o list.o network.o option.o parser.o simulator.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile include/simulator.h

all: $(OBJDIR) $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(COMMON) $^ -o $@ -lm

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $(COMMON) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJS) $(EXEC)
