CFLAGS = -std=c99 -Wextra -Wall -Wpedantic
LIBS = -lm
dep_headers = ../../tt.h ../../nn.h
dep_srcs = ../../tt.c ../../nn.c

all: data.exe train.exe test.exe

data.exe: data.c $(dep_srcs) $(dep_headers)
	gcc $(CFLAGS) -o data.exe data.c $(dep_srcs) $(LIBS)

train.exe: train.c cnn.h $(dep_srcs) $(dep_headers)
	gcc $(CFLAGS) -o train.exe train.c $(dep_srcs) $(LIBS)

test.exe: test.c cnn.h $(dep_srcs) $(dep_headers)
	gcc $(CFLAGS) -o test.exe test.c $(dep_srcs) $(LIBS)