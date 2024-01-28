#include	<stdlib.h>
#include	<time.h>
#include	<string.h>
#include	<assert.h>
#include	<stdio.h>
#include	<math.h>
#include	"tt.h"

typedef struct {
	TTTensor** tensors;
	int tensorsCnt;
} TensorAllocator;

static TensorAllocator* allocators = NULL;
static int allocatorsCnt = 0;

static float randomf();

void ttInitialize() {
	ttSeed(time(NULL));
	ttPushTensorAllocator();
}

void ttTerminate() {
	while(allocatorsCnt > 0)
		ttPopTensorAllocator();
}

static void* memdup(void* p, size_t size) {
	void* r = malloc(size);

	return memcpy(r, p, size);
}

int ttShapeToElnum(int* shape, int dim) {
	int elnum = 1;

	for(int i = 0; i < dim; i++)
		elnum *= shape[i];

	return elnum;
}

int ttIndexNDTo1D(int* shape, int dim, int* indices, int indicesCnt) {
	assert(indicesCnt <= dim);

	int base = 1;
	int index = 0;

	for(int d = dim - 1; d >= 0; d--) {
		if(d < indicesCnt) {
			assert(indices[d] < shape[d]);
			index += indices[d] * base;
		}
		
		base *= shape[d];
	}

	return index;
}

int* ttDupShape(TTTensor* tensor) {
	return memdup(tensor->shape, sizeof(int) * tensor->dim);
}

TTTensor** ttNewTensorArray(int count) {
	return (TTTensor**)malloc(sizeof(TTTensor*) * count);
}

TTTensor** ttTensorArrayAppend(TTTensor** tensors, int* count, TTTensor* tensor) {
	(*count)++;
	tensors = (TTTensor**)realloc(tensors, sizeof(TTTensor*) * (*count));
	tensors[*count - 1] = tensor;
	return tensors;
}

static TTTensor* newUninitDataTensor(int* shape, int dim) {
	TTTensor* tensor = (TTTensor*)malloc(sizeof(TTTensor));
	TensorAllocator* allocator = &allocators[allocatorsCnt - 1];
	int elnum = ttShapeToElnum(shape, dim);

	tensor->data = (float*)malloc(sizeof(float) * elnum);
	tensor->grad = (float*)calloc(elnum, sizeof(float));

	tensor->shape = (int*)malloc(sizeof(int) * dim);
	memcpy(tensor->shape, shape, sizeof(int) * dim);
	tensor->dim = dim;
	tensor->elnum = elnum;

	tensor->inGraph = false;
	tensor->_parentsCnt = 0;
	tensor->children = NULL;
	tensor->childrenCnt = 0;
	tensor->_backwardFunc = NULL;
	tensor->_visited = false;
	tensor->_deleteFunc = NULL;

	allocator->tensors = ttTensorArrayAppend(allocator->tensors, &(allocator->tensorsCnt), tensor);
	return tensor;
}

TTTensor* ttNewTensor(float* data, int* shape, int dim) {
	TTTensor* tensor = newUninitDataTensor(shape, dim);
	
	memcpy(tensor->data, data, sizeof(float) * tensor->elnum);
	return tensor;
}

TTTensor* ttNewZeroTensor(int* shape, int dim) {
	TTTensor* tensor = newUninitDataTensor(shape, dim);
	
	memset(tensor->data, 0, sizeof(float) * tensor->elnum);
	return tensor;
}

TTTensor* ttNewRandomRangeTensor(int* shape, int dim, float k) {
	TTTensor* tensor = newUninitDataTensor(shape, dim);

	for(int i = 0; i < tensor->elnum; i++)
		tensor->data[i] = randomf() * k * 2 - k;

	return tensor;
}

TTTensor* ttNewRandomTensor(int* shape, int dim) {
	return ttNewRandomRangeTensor(shape, dim, 0.1);
}

static void deleteTensor(TTTensor* tensor) {
	if(tensor->_deleteFunc)
		tensor->_deleteFunc(tensor);

	free(tensor->data);
	free(tensor->grad);
	free(tensor->shape);
	free(tensor->children);
	free(tensor);
}

void ttPushTensorAllocator() {
	allocatorsCnt++;
	allocators = (TensorAllocator*)realloc(
		allocators, sizeof(TensorAllocator) * allocatorsCnt
	);
	allocators[allocatorsCnt - 1] = (TensorAllocator){NULL, 0};
}

void ttPopTensorAllocator() {
	for(int i = 0; i < allocators[allocatorsCnt - 1].tensorsCnt; i++)
		deleteTensor(allocators[allocatorsCnt - 1].tensors[i]);

	free(allocators[allocatorsCnt - 1].tensors);
	allocatorsCnt--;
	allocators = (TensorAllocator*)realloc(allocators, allocatorsCnt);
}

void ttPrintTensor(TTTensor* tensor) {
	printf("data: [");

	for(int i = 0; i < tensor->elnum; i++)
		printf("%.5f, ", tensor->data[i]);

	printf("], grad: [");

	for(int i = 0; i < tensor->elnum; i++)
		printf("%.5f, ", tensor->grad[i]);

	printf("], shape: (");

	for(int i = 0; i < tensor->dim; i++)
		printf("%d, ", tensor->shape[i]);

	printf(")\n");
}

void ttPrintTensorShape(TTTensor* tensor) {
	printf("shape: (");

	for(int i = 0; i < tensor->dim; i++)
		printf("%d, ", tensor->shape[i]);

	printf(")\n");
}

static TTTensor** buildTopo(TTTensor* tensor, TTTensor** topo, int* topoLen) {
	if(!tensor->_visited) {
		tensor->_visited = true;

		for(int i = 0; i < tensor->childrenCnt; i++)
			topo = buildTopo(tensor->children[i], topo, topoLen);

		topo = ttTensorArrayAppend(topo, topoLen, tensor);
	}

	return topo;
}

void ttBackward(TTTensor* tensor) {
	for(int i = 0; i < tensor->elnum; i++)
		tensor->grad[i] = 1;

	TTTensor** topo = NULL;
	int topoLen = 0;

	topo = buildTopo(tensor, topo, &topoLen);

	for(int i = topoLen - 1; i >= 0; i--) {
		if(topo[i]->_backwardFunc != NULL)
			topo[i]->_backwardFunc(topo[i]);

		topo[i]->_visited = false;

		for(int j = 0; j < topo[i]->childrenCnt; j++) {
			topo[i]->children[j]->_parentsCnt -= 1;
			topo[i]->children[j]->inGraph = topo[i]->children[j]->_parentsCnt + topo[i]->children[j]->childrenCnt;
		}

		topo[i]->childrenCnt = 0;
		topo[i]->inGraph = topo[i]->_parentsCnt + topo[i]->childrenCnt;
	}

	free(topo);
}

bool ttIsSameShape(TTTensor* a, TTTensor* b) {
	if(a->dim != b->dim)
		return false;

	for(int i = 0; i < a->dim; i++) {
		if(a->shape[i] != b->shape[i])
			return false;
	}

	return true;
}

static void appendChild(TTTensor* tensor, TTTensor* child) {
	tensor->childrenCnt += 1;
	tensor->children = (TTTensor**)realloc(
		tensor->children,
		sizeof(TTTensor*) * tensor->childrenCnt
	);
	tensor->children[tensor->childrenCnt - 1] = child;
	child->_parentsCnt += 1;
	child->inGraph = true;
	tensor->inGraph = true;
}

static void addBackward(TTTensor* out) {
	TTTensor* a = out->children[0];
	TTTensor* b = out->children[1];

	for(int i = 0; i < out->elnum; i++) {
		a->grad[i] += out->grad[i];
		b->grad[i] += out->grad[i];
	}
}

TTTensor* ttAdd(TTTensor* a, TTTensor* b) {
	assert(ttIsSameShape(a, b));
	
	TTTensor* out = ttNewTensor(a->data, a->shape, a->dim);
	
	for(int i = 0; i < a->elnum; i++)
		out->data[i] += b->data[i];

	appendChild(out, a);
	appendChild(out, b);
	out->_backwardFunc = addBackward;
	return out;
}

static void subBackward(TTTensor* out) {
	TTTensor* a = out->children[0];
	TTTensor* b = out->children[1];

	for(int i = 0; i < a->elnum; i++) {
		a->grad[i] += out->grad[i];
		b->grad[i] -= out->grad[i];
	}
}

TTTensor* ttSub(TTTensor* a, TTTensor* b) {
	assert(ttIsSameShape(a, b));

	TTTensor* out = ttNewTensor(a->data, a->shape, a->dim);

	for(int i = 0; i < a->elnum; i++)
		out->data[i] -= b->data[i];

	appendChild(out, a);
	appendChild(out, b);
	out->_backwardFunc = subBackward;
	return out;
}

static void mulBackward(TTTensor* out) {
	TTTensor* a = out->children[0];
	TTTensor* b = out->children[1];

	for(int i = 0; i < a->elnum; i++) {
		a->grad[i] += b->data[i] * out->grad[i];
		b->grad[i] += a->data[i] * out->grad[i];
	}
}

TTTensor* ttMul(TTTensor* a, TTTensor* b) {
	assert(ttIsSameShape(a, b));

	TTTensor* out = ttNewTensor(a->data, a->shape, a->dim);

	for(int i = 0; i < a->elnum; i++)
		out->data[i] *= b->data[i];

	appendChild(out, a);
	appendChild(out, b);
	out->_backwardFunc = mulBackward;
	return out;
}

static void mulSBackward(TTTensor* out) {
	TTTensor* a = out->children[0];
	float scalar = out->data[0] / a->data[0];

	for(int i = 0; i < a->elnum; i++)
		a->grad[i] += scalar * out->grad[i];
}

TTTensor* ttMulS(TTTensor* a, float scalar) {
	TTTensor* out = ttNewTensor(a->data, a->shape, a->dim);

	for(int i = 0; i < a->elnum; i++)
		out->data[i] *= scalar;

	appendChild(out, a);
	out->_backwardFunc = mulSBackward;
	return out;
}

static void divBackward(TTTensor* out) {
	TTTensor* a = out->children[0];
	TTTensor* b = out->children[1];

	for(int i = 0; i < a->elnum; i++) {
		a->grad[i] += 1.0 / b->data[i] * out->grad[i];
		b->grad[i] += -a->data[i] / (b->data[i] * b->data[i]) * out->grad[i];
	}
}

TTTensor* ttDiv(TTTensor* a, TTTensor* b) {
	assert(ttIsSameShape(a, b));

	TTTensor* out = newUninitDataTensor(a->shape, a->dim);

	for(int i = 0; i < a->elnum; i++)
		out->data[i] = a->data[i] / b->data[i];

	appendChild(out, a);
	appendChild(out, b);
	out->_backwardFunc = divBackward;
	return out;
}

static void divSBackward(TTTensor* out) {
	TTTensor* a = out->children[0];
	float scalar = a->data[0] / out->data[0];

	for(int i = 0; i < a->elnum; i++)
		a->grad[i] = 1.0 / scalar * out->grad[i];
}

TTTensor* ttDivS(TTTensor* a, float scalar) {
	TTTensor* out = newUninitDataTensor(a->shape, a->dim);

	for(int i = 0; i < a->elnum; i++)
		out->data[i] = a->data[i] / scalar;

	appendChild(out, a);
	out->_backwardFunc = divSBackward;
	return out;
}

TTTensor* ttNeg(TTTensor* tensor) {
	return ttMulS(tensor, -1);
}

static void matMulBackward(TTTensor* out) {
	TTTensor* a = out->children[0];
	TTTensor* b = out->children[1];

	for(int i = 0; i < a->shape[0]; i++) {
		for(int j = 0; j < a->shape[1]; j++) {
			for(int k = 0; k < b->shape[1]; k++)
				a->grad[i * a->shape[1] + j] += out->grad[i * out->shape[1] + k] * b->data[j * b->shape[1] + k];
		}
	}

	for(int i = 0; i < b->shape[0]; i++) {
		for(int j = 0; j < b->shape[1]; j++) {
			for(int k = 0; k < a->shape[0]; k++)
				b->grad[i * b->shape[1] + j] += out->grad[k * out->shape[1] + j] * a->data[k * a->shape[1] + i];
		}
	}
}

TTTensor* ttMatMul(TTTensor* a, TTTensor* b) {
	assert(a->dim >= 2 && b->dim == 2 && a->shape[a->dim - 1] == b->shape[0]);

	int shape[2] = {a->shape[0], b->shape[1]};

	TTTensor* out = ttNewZeroTensor(shape, 2);

	for(int i = 0; i < out->shape[0]; i++) {
		for(int j = 0; j < out->shape[1]; j++) {
			for(int k = 0; k < b->shape[0]; k++)
				out->data[i * out->shape[1] + j] += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
		}
	}

	appendChild(out, a);
	appendChild(out, b);
	out->_backwardFunc = matMulBackward;
	return out;
}

static void sigmoidBackward(TTTensor* out) {
	TTTensor* x = out->children[0];

	for(int i = 0; i < out->elnum; i++)
		x->grad[i] += out->grad[i] * (out->data[i] * (1 - out->data[i]));
}

TTTensor* ttSigmoid(TTTensor* tensor) {
	TTTensor* out = newUninitDataTensor(tensor->shape, tensor->dim);

	for(int i = 0; i < out->elnum; i++)
		out->data[i] = 1 / (1 + exp(-tensor->data[i]));

	appendChild(out, tensor);
	out->_backwardFunc = sigmoidBackward;
	return out;
}

static void reluBackward(TTTensor* out) {
	TTTensor* x = out->children[0];

	for(int i = 0; i < out->elnum; i++)
		x->grad[i] += out->grad[i] * ((x->data[i] > 0) ? 1 : 0);
}

TTTensor* ttRelu(TTTensor* tensor) {
	TTTensor* out = newUninitDataTensor(tensor->shape, tensor->dim);

	for(int i = 0; i < out->elnum; i++)
		out->data[i] = (tensor->data[i] > 0) ? tensor->data[i] : 0;

	appendChild(out, tensor);
	out->_backwardFunc = reluBackward;
	return out;
}

static void getBackward(TTTensor* out) {
	TTTensor* in = out->children[0];
	int* indexMap = (int*)out->_pdata;

	for(int i = 0; i < out->elnum; i++)
		in->grad[indexMap[i]] += out->grad[i];
}

static int clampIndex(int index, int len) {
	if(index < 0)
		index = len + index;

	if(index < 0)
		return 0;
	else if(index > len)
		return len;

	return index;
}

static void normalizeIndexer(TTIndexer* indexer, int len) {
	indexer->_at = clampIndex(indexer->_at, len);
	
	if(indexer->_isSlice) {
		if(indexer->_toEnd)
			indexer->_to = len;
		else
			indexer->_to = clampIndex(indexer->_to, len);
	}
}

static void getDelete(TTTensor* tensor) {
	free(tensor->_pdata);
}

TTTensor* ttGet(TTTensor* tensor, TTIndexer* indexers, int indexersCnt) {
	assert(indexersCnt <= tensor->dim);

	int dim = tensor->dim; // new dimensions
	int* shape; // new shape
	int* oriDims; // same length as shape, map the dimension to the original tensor
	TTTensor* out;

	// padding indexers as long as tensor->dim
	indexers = memdup(indexers, sizeof(TTIndexer) * indexersCnt); 
	indexers = (TTIndexer*)realloc(indexers, sizeof(TTIndexer) * tensor->dim);

	for(int i = 0; i < tensor->dim; i++) {
		if(i >= indexersCnt)
			indexers[i] = ttAll();
		else if(!indexers[i]._isSlice)
			dim--;
	}

	shape = (int*)malloc(sizeof(int) * dim);
	oriDims = (int*)malloc(sizeof(int) * dim); // [output dimension] => input dimension

	int d = 0;

	for(int i = 0; i < tensor->dim; i++) {
		if(indexers[i]._isSlice) {
			normalizeIndexer(&indexers[i], tensor->shape[i]);
			assert(indexers[i]._to > indexers[i]._at);
			shape[d] = indexers[i]._to - indexers[i]._at;
			oriDims[d] = i;
			d++;
		}
	}

	int* inIndex = (int*)malloc(sizeof(int) * tensor->dim);

	out = newUninitDataTensor(shape, dim);

	int* indexMap = (int*)malloc(sizeof(int) * out->elnum);

	for(int i = 0; i < tensor->dim; i++)
		inIndex[i] = indexers[i]._at;

	for(int i = 0; i < out->elnum; i++) {
		indexMap[i] = ttIndexNDTo1D(tensor->shape, tensor->dim, inIndex, tensor->dim);
		out->data[i] = tensor->data[indexMap[i]];
		

		if(i + 1 == out->elnum)
			break;

		inIndex[oriDims[dim - 1]] += 1;

		for(int j = dim - 1; j >= 0; j--) {
			if(inIndex[oriDims[j]] == indexers[oriDims[j]]._to) {
				inIndex[oriDims[j]] = indexers[oriDims[j]]._at;
				inIndex[oriDims[j - 1]] += 1;
			}
		}
	}

	out->_pdata = indexMap;
	out->_backwardFunc = getBackward;
	appendChild(out, tensor);
	out->_deleteFunc = getDelete;
	free(shape);
	free(oriDims);
	free(inIndex);
	free(indexers);
	return out;
}

TTIndexer ttAt(int index) {
	return (TTIndexer){._at = index, ._isSlice = false};
}

TTIndexer ttSlice(int from, int to) {
	return (TTIndexer){._at = from, ._to = to, ._isSlice = true, ._toEnd = false};
}

TTIndexer ttFrom(int from) {
	return (TTIndexer){._at = from, ._isSlice = true, ._toEnd = true};
}

TTIndexer ttTo(int to) {
	return (TTIndexer){._at = 0, ._to = to, ._isSlice = true, ._toEnd = false};
}

TTIndexer ttAll() {
	return (TTIndexer){._at = 0, ._isSlice = true, ._toEnd = true};
}

static void stackBackward(TTTensor* tensor) {
	for(int i = 0; i < tensor->childrenCnt; i++) {
		for(int j = 0; j < tensor->children[0]->elnum; j++)
			tensor->children[i]->grad[j] += tensor->grad[i * tensor->children[0]->elnum + j];
	}
}

TTTensor* ttStack(TTTensor** tensors, int tensorsCnt) {
	for(int i = 0; i < tensorsCnt - 1; i++)
		assert(ttIsSameShape(tensors[i], tensors[i + 1]));

	int* shape = (int*)malloc(sizeof(int) * (tensors[0]->dim + 1));

	memcpy(&shape[1], tensors[0]->shape, sizeof(int) * tensors[0]->dim);
	shape[0] = tensorsCnt;

	TTTensor* out = newUninitDataTensor(shape, tensors[0]->dim + 1);

	for(int i = 0; i < tensorsCnt; i++) {
		memcpy(&out->data[tensors[0]->elnum * i], tensors[i]->data, sizeof(float) * tensors[0]->elnum);
		appendChild(out, tensors[i]);
	}

	out->_backwardFunc = stackBackward;
	free(shape);
	return out;
}

static void reshapeBackward(TTTensor* out) {
	TTTensor* in = out->children[0];

	for(int i = 0; i < out->elnum; i++)
		in->grad[i] += out->grad[i];
}

TTTensor* ttReshape(TTTensor* tensor, int* shape, int dim) {
	assert(ttShapeToElnum(shape, dim) == tensor->elnum);

	TTTensor* out = ttNewTensor(tensor->data, shape, dim);

	appendChild(out, tensor);
	out->_backwardFunc = reshapeBackward;
	return out;
}

static void sumBackward(TTTensor* out) {
	TTTensor* in = out->children[0];

	for(int i = 0; i < in->elnum; i++)
		in->grad[i] += out->grad[0];
}

TTTensor* ttSum(TTTensor* tensor) {
	float sum = 0;
	TTTensor* out;

	for(int i = 0; i < tensor->elnum; i++)
		sum += tensor->data[i];

	out = ttNewTensor((float[]){sum}, NULL, 0);
	appendChild(out, tensor);
	out->_backwardFunc = sumBackward;
	return out;
}

static void maxBackward(TTTensor* out) {
	*((float*)out->_pdata) += out->grad[0];
}

TTTensor* ttMax(TTTensor* tensor) {
	float max = tensor->data[0];
	float* maxGrad = tensor->grad;
	TTTensor* out;

	for(int i = 0; i < tensor->elnum; i++) {
		if(tensor->data[i] > max) {
			max = tensor->data[i];
			maxGrad = tensor->grad + i;
		}
	}

	out = ttNewTensor((float[]){max}, NULL, 0);
	appendChild(out, tensor);
	out->_pdata = maxGrad;
	out->_backwardFunc = maxBackward;
	return out;
}

static void expBackward(TTTensor* out) {
	TTTensor* in = out->children[0];

	for(int i = 0; i < out->elnum; i++)
		in->grad[i] += out->data[i] * out->grad[i];
}

TTTensor* ttExp(TTTensor* tensor) {
	TTTensor* out = newUninitDataTensor(tensor->shape, tensor->dim);

	for(int i = 0; i < out->elnum; i++)
		out->data[i] = exp(tensor->data[i]);

	appendChild(out, tensor);
	out->_backwardFunc = expBackward;
	return out;
}

static void logBackward(TTTensor* out) {
	TTTensor* in = out->children[0];

	for(int i = 0; i < out->elnum; i++)
		in->grad[i] += 1 / in->data[i] * out->grad[i];
}

TTTensor* ttLog(TTTensor* tensor) {
	TTTensor* out = newUninitDataTensor(tensor->shape, tensor->dim);

	for(int i = 0; i < out->elnum; i++)
		out->data[i] = log(tensor->data[i]);

	appendChild(out, tensor);
	out->_backwardFunc = logBackward;
	return out;
}

void ttSave(TTTensor* tensor, const char* path) {
	FILE* f = fopen(path, "wb");

	fwrite(&tensor->dim, sizeof(int), 1, f);
	fwrite(tensor->shape, sizeof(int), tensor->dim, f);
	fwrite(tensor->data, sizeof(float), tensor->elnum, f);
	fclose(f);
}

TTTensor* ttLoad(const char* path) {
	FILE* f = fopen(path, "rb");
	int dim;
	int* shape;
	TTTensor* tensor;

	fread(&dim, sizeof(int), 1, f);
	shape = (int*)malloc(sizeof(int) * dim);
	fread(shape, sizeof(int), dim, f);
	tensor = newUninitDataTensor(shape, dim);
	fread(tensor->data, sizeof(float), tensor->elnum, f);
	free(shape);
	fclose(f);
	return tensor;
}

/* 
   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

	 1. Redistributions of source code must retain the above copyright
		notice, this list of conditions and the following disclaimer.

	 2. Redistributions in binary form must reproduce the above copyright
		notice, this list of conditions and the following disclaimer in the
		documentation and/or other materials provided with the distribution.

	 3. The names of its contributors may not be used to endorse or promote 
		products derived from this software without specific prior written 
		permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// Following code is modified from the C implementation of Mersenne Twister
// pseudo number generater (MT19937) by Takuji Nishimura and Makoto Matsumoto.
// Use as alternative for stdlib rand().
// Reference: http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/MT2002/CODES/mt19937ar.c

static uint32_t mt[624];
static int mti = 624 + 1;

void ttSeed(uint32_t s) {
	mt[0]= s & 0xffffffffUL;

	for(mti = 1; mti < 624; mti++) {
		mt[mti] = (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
		mt[mti] &= 0xffffffffUL;
	}
}

static float randomf() {
	uint32_t y;
	static uint32_t mag01[2]={0x0UL, 0x9908b0dfUL};

	if(mti >= 624) {
		int kk;

		if(mti == 624 + 1)
			ttSeed(5489UL);

		for(kk = 0; kk < 624 - 397; kk++) {
			y = (mt[kk] & 0x80000000UL) | (mt[kk+1] & 0x7fffffffUL);
			mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}

		for(; kk < 624 - 1; kk++) {
			y = (mt[kk] & 0x80000000UL) | (mt[kk+1] & 0x7fffffffUL);
			mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 0x1UL];
		}

		y = (mt[624 - 1] & 0x80000000UL)|(mt[0] & 0x7fffffffUL);
		mt[624 - 1] = mt[397 - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];
		mti = 0;
	}
  
	y = mt[mti++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680UL;
	y ^= (y << 15) & 0xefc60000UL;
	y ^= (y >> 18);
	return y * (1.0 / 4294967295.0);
}