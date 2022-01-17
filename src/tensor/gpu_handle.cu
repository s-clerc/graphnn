#include "tensor/gpu_handle.h"
#include "util/gnn_macros.h"
#include "util/mem_holder.h"
#include "tbb/tbb.h"

namespace gnn
{

__global__ void SetupRandKernel(curandState_t *state, unsigned long long seed) 
{
    const unsigned int tidx = NUM_RND_THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;
    /* Each thread gets same seed, a different sequence number,
     no offset */
    curand_init(seed, tidx, 0, &state[tidx]);
}

void GpuHandle::Init(int dev_id, unsigned int _streamcnt)
{
	// My understanding is that according to https://github.com/DrTimothyAldenDavis/SuiteSparse/issues/72
	// in modern oneTBB, excluding the line below means that the system will automatically allocate threads
	// and since it is fixed to a constant, I don't think it does anything very important wrt to the other
	// libraries being used.
	// If needed, we could use https://oneapi-src.github.io/oneTBB/main/tbb_userguide/Migration_Guide/Task_Scheduler_Init.html
	// 's max concurrency to limit it to 4 I believe.
	//tbb::task_scheduler_init init(4);
	streamcnt = _streamcnt;
	cudaDeviceReset();
	cudaSetDevice(dev_id);

	cublashandles = new cublasHandle_t[streamcnt];
	cusparsehandles = new cusparseHandle_t[streamcnt];
	inUse = new bool[streamcnt];
	while (!resources.empty())
		resources.pop();
	for (unsigned int id = 0; id < streamcnt; ++id)
	{
		cublasCreate(&cublashandles[id]);	
		cusparseCreate(&cusparsehandles[id]);
		inUse[id] = false;
		resources.push(id);
	}
	cudaStreamCreate(&cudaRandStream);

	curandCreateGenerator(&curandgenerator, CURAND_RNG_PSEUDO_DEFAULT);
	
	curandSetPseudoRandomGeneratorSeed(curandgenerator, time(NULL));
	
    cudaMalloc((void **)&devRandStates, NUM_RND_STREAMS * sizeof(curandState_t));
	SetupRandKernel<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(devRandStates, 1 + time(NULL)*2);
}

GpuContext GpuHandle::AquireCtx()
{
	r_loc.lock();
	ASSERT(resources.size(), "running out of gpu resources");

	int cur_pos = resources.front();
	resources.pop();

	r_loc.unlock();

	ASSERT(!inUse[cur_pos], "logic error: in-use resource is found available");
	inUse[cur_pos] = true;	
	cublasSetStream(cublashandles[cur_pos], cudaStreamPerThread);
	cusparseSetStream(cusparsehandles[cur_pos], cudaStreamPerThread);
	return GpuContext(cur_pos, cublashandles[cur_pos], cusparsehandles[cur_pos]);
}

void GpuHandle::ReleaseCtx(const GpuContext& ctx)
{
	r_loc.lock();
	resources.push(ctx.id);
	ASSERT(inUse[ctx.id], "logic error: in-use resource is not recorded, or you are releasing same resource multiple times");
	inUse[ctx.id] = false;
	r_loc.unlock();	
}

void GpuHandle::Destroy()
{
	cudaDeviceSynchronize();
	cudaStreamDestroy(cudaRandStream);
	for (unsigned int id = 0; id < streamcnt; ++id)
	{
		cublasDestroy_v2(cublashandles[id]);
		cusparseDestroy(cusparsehandles[id]);
	}
	delete[] cublashandles;
	delete[] cusparsehandles;
	delete[] inUse;
	curandDestroyGenerator(curandgenerator);
    cudaFree(devRandStates);
	streamcnt = 0U;
}

curandState_t* GpuHandle::devRandStates = NULL;
cublasHandle_t* GpuHandle::cublashandles = NULL;
cusparseHandle_t* GpuHandle::cusparsehandles = NULL;
curandGenerator_t GpuHandle::curandgenerator;
unsigned int GpuHandle::streamcnt = 1U;
std::queue< int > GpuHandle::resources;
std::mutex GpuHandle::r_loc;
std::mutex GpuHandle::rand_lock;
bool* GpuHandle::inUse = NULL;
cudaStream_t GpuHandle::cudaRandStream;

}