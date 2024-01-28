#ifndef	TT_NN_H
#define	TT_NN_H

#include	"tt.h"

void ttZeroGrad(TTTensor** tensors, int count);
void ttOptimize(TTTensor** tensors, int count, float lr);

typedef struct {
	TTTensor* W;
	TTTensor* b;
	int in;
	int out;
} TTLinearParams;

TTLinearParams ttNewLinear(int in, int out);
TTTensor* ttLinear(TTLinearParams params, TTTensor* x);

typedef struct {
	TTTensor* W;
	TTTensor* b;
	int in;
	int out;
	int kernel;
} TTConv2DParams;

TTConv2DParams ttNewConv2D(int in, int out, int kernel);
TTTensor* ttConv2D(TTConv2DParams params, TTTensor* x);

TTTensor* ttMaxPool2D(TTTensor* x, int kernel);

TTTensor* ttRepeat(TTTensor* tensor, int* shape, int dim);

TTTensor* ttSoftmax(TTTensor* x);
TTTensor* ttCrossEntropy(TTTensor* input, TTTensor* target);

TTTensor* ttMSE(TTTensor* input, TTTensor* target);

#endif