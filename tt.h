#ifndef	TT_H
#define	TT_H

#include	<stdbool.h>
#include	<stdint.h>

typedef struct _TTTensor {
	// Can be modified only if !inGraph, otherwise read-only
	float* data;
	float* grad;

	// Read-only fields
	int* shape;
	int dim;
	int elnum;

	struct _TTTensor** children;
	int childrenCnt;
	bool inGraph;

	// Private fields
	int _parentsCnt;
	void (*_backwardFunc)(struct _TTTensor*);
	bool _visited;

	void* _pdata; // Reserved for some operations
	void (*_deleteFunc)(struct _TTTensor*);
} TTTensor;

typedef struct {
	int _at;
	int _to;
	bool _isSlice;
	bool _toEnd;
} TTIndexer; // Don't create manually, use the indexer functions.

// Initialization and termination, memory allocation.
void ttInitialize();
void ttTerminate();
void ttPopTensorAllocator();
void ttPushTensorAllocator();
void ttSeed(uint32_t seed);

// Tensor allocator works like a stack like mechanism, calling pop()
// will free the memory of tensors allocated since the last push() call.

// Tensor creation
TTTensor* ttNewTensor(float* data, int* shape, int dim);
TTTensor* ttNewZeroTensor(int* shape, int dim);
TTTensor* ttNewRandomTensor(int* shape, int dim);
TTTensor* ttNewRandomRangeTensor(int* shape, int dim, float k);

// Operations (all operation returns a new tensor)
TTTensor* ttAdd(TTTensor* a, TTTensor* b);
TTTensor* ttSub(TTTensor* a, TTTensor* b);
TTTensor* ttMul(TTTensor* a, TTTensor* b);
TTTensor* ttMulS(TTTensor* a, float scalar);
TTTensor* ttDiv(TTTensor* a, TTTensor* b);
TTTensor* ttDivS(TTTensor* a, float scalar);
TTTensor* ttNeg(TTTensor* tensor);
TTTensor* ttExp(TTTensor* tensor);
TTTensor* ttLog(TTTensor* tensor);
TTTensor* ttMatMul(TTTensor* a, TTTensor* b);
TTTensor* ttSigmoid(TTTensor* tensor);
TTTensor* ttRelu(TTTensor* tensor);

TTTensor* ttStack(TTTensor** tensors, int tensorsCnt);
TTTensor* ttReshape(TTTensor* tensor, int* shape, int dim);
TTTensor* ttSum(TTTensor* tensor);
TTTensor* ttMax(TTTensor* tensor);

TTTensor* ttGet(TTTensor* tensor, TTIndexer* indexers, int indexersCnt);

// Indexing and slicing, python like syntax.
TTIndexer ttAt(int index);           // [index]
TTIndexer ttSlice(int from, int to); // [from:to]
TTIndexer ttFrom(int from);          // [from:]
TTIndexer ttTo(int to);              // [:to]
TTIndexer ttAll();                   // [:]

// E.g. x[0,2:-1,3:] => ttGet(x, (TTIndexer[]){ttAt(0), ttSlice(2, -1), ttFrom(3)}, 3)

// Auto-grad (back propagation)
void ttBackward(TTTensor* tensor);

// Tensor saving to and loading from file.
void ttSave(TTTensor* tensor, const char* path);
TTTensor* ttLoad(const char* path);

// Utility functions
TTTensor** ttNewTensorArray(int count);
TTTensor** ttTensorArrayAppend(TTTensor** tensors, int* count, TTTensor* tensor);
int ttShapeToElnum(int* shape, int dim);
int ttIndexNDTo1D(int* shape, int dim, int* indices, int indicesCnt);
bool ttIsSameShape(TTTensor* a, TTTensor* b);
int* ttDupShape(TTTensor* tensor);
void ttPrintTensor(TTTensor* tensor);
void ttPrintTensorShape(TTTensor* tensor);

#endif