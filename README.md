# tensortoy

A simple tensor operation and auto-grad (back propagation) engine implement in C99 with C standard library only. Inspired by [micrograd](https://github.com/karpathy/micrograd).

### Usage

No other dependencies needed, just include `tt.h` in your program and compile along with `tt.c`. (And optionally `nn.h` and `nn.c`)

Docs can be found in the header files, a CNN MNIST training example is included.