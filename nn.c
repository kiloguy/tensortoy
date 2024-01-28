#include	<math.h>
#include	<assert.h>
#include	<string.h>
#include	<stdlib.h>
#include	"nn.h"

void ttZeroGrad(TTTensor** tensors, int count) {
	for(int i = 0; i < count; i++)
		memset(tensors[i]->grad, 0, sizeof(float) * tensors[i]->elnum);
}

void ttOptimize(TTTensor** tensors, int count, float lr) {
	for(int i = 0; i < count; i++) {
		assert(!tensors[i]->inGraph);

		for(int j = 0; j < tensors[i]->elnum; j++)
			tensors[i]->data[j] -= tensors[i]->grad[j] * lr;
	}
}

TTLinearParams ttNewLinear(int in, int out) {
	return (TTLinearParams){
		.W = ttNewRandomRangeTensor((int[]){in, out}, 2, sqrt(1.0 / in)),
		.b = ttNewRandomRangeTensor((int[]){1, out}, 2, sqrt(1.0 / in)),
		.in = in,
		.out = out
	};
}

TTTensor* ttLinear(TTLinearParams params, TTTensor* x) {
	assert(x->dim >= 1);

	int m = x->elnum / params.in;
	TTTensor* flat = ttReshape(x, (int[]){m, 1, params.in}, 3);
	TTTensor** tensors = ttNewTensorArray(m);

	for(int i = 0; i < m; i++)
		tensors[i] = ttAdd(ttMatMul(ttGet(flat, (TTIndexer[]){ttAt(i)}, 1), params.W), params.b);

	TTTensor* out = ttStack(tensors, m);
	int* shape = ttDupShape(x);

	shape[x->dim - 1] = params.out;
	out = ttReshape(out, shape, x->dim);
	free(tensors);
	free(shape);
	return out;
}

TTConv2DParams ttNewConv2D(int in, int out, int kernel) {
	return (TTConv2DParams){
		.W = ttNewRandomRangeTensor((int[]){out, in, kernel, kernel}, 4, sqrt(1.0 / (in * kernel * kernel))),
		.b = ttNewRandomRangeTensor((int[]){out}, 1, sqrt(1.0 / (in * kernel * kernel))),
		.in = in,
		.out = out,
		.kernel = kernel
	};
}

TTTensor* ttConv2D(TTConv2DParams params, TTTensor* x) {
	assert(x->dim == 3);


	int outH = x->shape[1] - params.kernel + 1;
	int outW = x->shape[2] - params.kernel + 1;
	TTTensor** row = ttNewTensorArray(outW);
	TTTensor** rows = ttNewTensorArray(outH);
	TTTensor** faces = ttNewTensorArray(params.out);
	TTTensor** masks = ttNewTensorArray(params.out);
	TTTensor** bs = ttNewTensorArray(params.out);
	TTTensor* out;

	for(int i = 0; i < params.out; i++) {
		masks[i] = ttGet(params.W, (TTIndexer[]){ttAt(i)}, 1);
		bs[i] = ttGet(params.b, (TTIndexer[]){ttAt(i)}, 1);
	}

	for(int i = 0; i < params.out; i++) {
		for(int j = 0; j < outH; j++) {
			for(int k = 0; k < outW; k++) {
				TTTensor* local = ttGet(x, (TTIndexer[]){ttAll(), ttSlice(j, j + params.kernel), ttSlice(k, k + params.kernel)}, 3);
				
				row[k] = ttAdd(ttSum(ttMul(local, masks[i])), bs[i]);
			}

			rows[j] = ttStack(row, outW);
		}

		faces[i] = ttStack(rows, outH);
	}

	out = ttStack(faces, params.out);
	free(row);
	free(rows);
	free(faces);
	free(masks);
	free(bs);
	return out;
}

TTTensor* ttMaxPool2D(TTTensor* x, int kernel) {
	assert(x->dim == 3);

	int outH = x->shape[1] - kernel + 1;
	int outW = x->shape[2] - kernel + 1;
	TTTensor** row = ttNewTensorArray(outW);
	TTTensor** rows = ttNewTensorArray(outH);
	TTTensor** faces = ttNewTensorArray(x->shape[0]);
	TTTensor* out;

	for(int i = 0; i < x->shape[0]; i++) {
		for(int j = 0; j < outH; j++) {
			for(int k = 0; k < outW; k++)
				row[k] = ttMax(ttGet(x, (TTIndexer[]){ttAt(i), ttSlice(j, j + kernel), ttSlice(k, k + kernel)}, 3));

			rows[j] = ttStack(row, outW);
		}

		faces[i] = ttStack(rows, outH);
	}

	out = ttStack(faces, x->shape[0]);
	free(faces);
	free(row);
	free(rows);
	return out;
}

TTTensor* ttRepeat(TTTensor* tensor, int* shape, int dim) {
	int cnt = ttShapeToElnum(shape, dim);
	TTTensor** tensors = ttNewTensorArray(cnt);
	int* newShape = (int*)malloc(sizeof(int) * (dim + tensor->dim));

	for(int i = 0; i < cnt; i++)
		tensors[i] = tensor;

	memcpy(newShape, shape, sizeof(int) * dim);
	memcpy(newShape + dim, tensor->shape, sizeof(int) * tensor->dim);
	return ttReshape(ttStack(tensors, cnt), newShape, dim + tensor->dim);
}

TTTensor* ttSoftmax(TTTensor* x) {
	TTTensor* expX = ttExp(x);

	return ttDiv(expX, ttRepeat(ttSum(expX), x->shape, x->dim));
}

TTTensor* ttCrossEntropy(TTTensor* input, TTTensor* target) {
	assert(input->dim == 1 && target->dim == 1 && input->shape[0] == target->shape[0]);

	for(int i = 0; i < target->shape[0]; i++)
		assert(target->data[i] >= 0 && target->data[i] <= 1);

	return ttNeg(ttSum(ttMul(ttLog(ttSoftmax(input)), target)));
}

TTTensor* ttMSE(TTTensor* input, TTTensor* target) {
	assert(ttIsSameShape(input, target));

	TTTensor* diff = ttSub(input, target);

	return ttDivS(ttSum(ttMul(diff, diff)), diff->elnum);
}