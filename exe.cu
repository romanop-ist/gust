////////////////////
////	SHMEM	////
////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <vector>

#define DISJOINT 0
#define THREAD_TX_SHIFT 0
#define KERNEL_DURATION 5
#define PR_MAX_RWSET_SIZE 64

#include "pr-stm.cuh"
#include "pr-stm-internal.cuh"
#include "util.cuh"
#include <unistd.h>
#include "ycsb.cuh"

typedef struct Statistics_
{
	long long int total;
	long long int runtime;
	long long int commit;
	long int nbReadOnly;
	long int nbUpdates;	
} Statistics;

#define CUDA_CHECK_ERROR(func, msg) ({ \
	cudaError_t cudaError; \
	if (cudaSuccess != (cudaError = func)) { \
		fprintf(stderr, #func ": in " __FILE__ ":%i : " msg "\n   > %s\n", \
		__LINE__, cudaGetErrorString(cudaError)); \
    *((int*)0x0) = 0; /* exit(-1); */ \
	} \
  cudaError; \
})

using namespace bench_ycsb;

__global__ void client_kernel(
	uint dataSize, 
	int* data, 
	/*readSet* rs, writeSet* ws,*/
	char *txs,
	int batch_st,
	int batch_en,
	PR_globalKernelArgs,  
	Statistics* stats) 
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid & ((1 << THREAD_TX_SHIFT) - 1))
        return;
    size_t thread_id = batch_st + (tid >> THREAD_TX_SHIFT);

    if (thread_id >= batch_en)
        return;

    PR_enterKernel(tid);
	bool tx_success = false;

	YCSBTx tx = ((YCSBTx *)txs)[thread_id];

#if PERF_METRICS
	long long int start_time_commit, stop_time_commit;
	long long int start_time_tx, start_time_total, stop_time_tx;
#endif

#if PERF_METRICS
	start_time_total = clock64();
#endif

    while(!tx_success)
    {

#if PERF_METRICS
		start_time_tx = clock64();
#endif
        PR_txBegin();

        for (int i = 0; i < tx.request_cnt; i++)
        {
            YCSBReq &req = tx.requests[i];
            size_t idx = req.key;
            
            if (req.read)
            {
                PR_read(&data[idx]);
            }
            else
            {
				PR_read(&data[idx]);
				PR_write(&data[idx], tid);
            }

        }
#if PERF_METRICS
		start_time_commit = clock64();
#endif
		PR_txCommit();
		tx_success = true;
    }

#if PERF_METRICS
	stop_time_commit = clock64();
	stop_time_tx = clock64();

	stats[tid].total   		 += stop_time_tx - start_time_total;
	stats[tid].runtime 		 += stop_time_tx - start_time_tx;
	stats[tid].commit 		 += stop_time_commit - start_time_commit;			
#endif

	// exiting threads reset write status, so other threads that would conflict with it can ignore this
    PR_exitKernel();
}

// __host__ float parent_kernel(uint block_size,
// 								int txcnt, int batch_size, int valid_txn_bitoffset,
// 								char* txs,
// 								uint dataSize, int* data, 
// 								/*readSet* rs, writeSet* ws,*/
// 								PR_globalKernelArgs,
// 								Statistics* stats) {
	
// 	float tKernel_ms = 0.0, totT_ms = 0.0;
// 	cudaEvent_t start, stop;
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&stop);

//     std::vector<size_t> batches;
//     std::vector<cudaStream_t> streams;
  	
//   	for (size_t st = 0; st < txcnt; st += batch_size)
//     	batches.push_back(std::min<unsigned long>(batch_size, txcnt - st));
//     streams.resize(batches.size());

//     int batch_st = 0;
//     int batch_en = 0;

//     for (int i = 0; i < batches.size(); i++)
//     {
//         size_t batched_txnum = batches[i];
//         if (!batched_txnum)
//             continue;
//         batch_en = batch_st + batched_txnum;
//         size_t thread_num = batched_txnum << valid_txn_bitoffset;

//         dim3 num_blocks(thread_num / block_size + (thread_num % block_size == 0 ? 0 : 1), 1, 1);
//         dim3 num_threads(thread_num > block_size ? block_size : thread_num, 1, 1);

//         PR_blockNum = thread_num / block_size + (thread_num % block_size == 0 ? 0 : 1);
// 		PR_threadNum = thread_num > block_size ? block_size : thread_num;

//         cudaEventRecord(start);
//         PR_prepare_noCallback(&args);

// 		client_kernel<<<num_blocks, num_threads, 0, streams[i]>>>(
// 			dataSize, 
// 			data, /*rs, ws,*/
// 			txs,
// 			batch_st,
// 			batch_en,
// 			args.dev,
// 			stats);
        
//         PR_postrun_noCallback(&args);
//         cudaEventRecord(stop);
//         cudaEventSynchronize(stop);

//         cudaEventElapsedTime(&tKernel_ms, start, stop);
//         totT_ms += tKernel_ms;

//         PR_reduceCommitAborts<<<PR_blockNum, PR_threadNum, 0, PR_streams[PR_currentStream]>>>
// 		(0, PR_currentStream, args.dev, sumNbCommits, sumNbAborts);

//         batch_st = batch_en;
//     }
//     for (auto &stream : streams)
//     	cudaStreamSynchronize(stream);

// 	return totT_ms;
// }

void getKernelOutput(Statistics *h_times, uint threadNum, int peak_clk, float totT_ms, uint64_t nbCommits, uint64_t nbAborts, uint verbose)
{
  	double avg_total=0, avg_runtime=0, avg_commit=0, avg_waste=0;
  	long int totReads=0, totUpdates=0;
	
	//long int nbAborts = *PR_sumNbAborts;
	//long int commits;

	for(int i=0; i<threadNum; i++)
	{
		avg_total   += h_times[i].total;
		avg_runtime += h_times[i].runtime;
		avg_commit 	+= h_times[i].commit;
		avg_waste   += h_times[i].total - h_times[i].runtime;

		totReads 	+= h_times[i].nbReadOnly;
		totUpdates	+= h_times[i].nbUpdates;
	}
	
	//nbCommits = totReads + totUpdates;
	long int denom = nbCommits*peak_clk;
	avg_total	/= denom;
	avg_runtime	/= denom;
	avg_commit 	/= denom;
	avg_waste   /= denom;

	float rt_commit=0.0;
	rt_commit	=	avg_commit / avg_runtime;

	//printf("nbCommits: %d\n", nbCommits);
	
	if(verbose)
		printf("AbortPercent\t%f %%\nThroughtput\t%f\n\nTotal\t\t%f\nRuntime\t\t%f\nCommit\t\t%f\t%.2f%%\nWaste\t\t%f\n",
			(float)nbAborts/(nbAborts+nbCommits)*100.0,
			nbCommits/totT_ms*1000.0,
			avg_total,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			avg_waste
			);
	else
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\n", 
			(float)nbAborts/(nbAborts+nbCommits)*100.0,
			nbCommits/totT_ms*1000.0,
			avg_total,
			avg_runtime,
			avg_commit,
			rt_commit*100.0,
			avg_waste
			);
}

void test_fine_grain_offloading(YCSBTx *txdata, int dataSize, int txCount, int valid_txn_bitoffset, int block_size, int batch_size, int verbose)
{
///////////////

	int *h_data, *d_data;

	Statistics *h_stats, *d_stats;

  	int peak_clk=1;
	cudaError_t err = cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);

	h_stats = (Statistics*) calloc(txCount,sizeof(Statistics));
	//h_stats = (Statistics*)calloc(1,sizeof(Statistics));

	h_data = (int*)calloc(dataSize,sizeof(int));
	
	//Allocate memory in the device
	cudaError_t result;
	PR_init(1);
	pr_tx_args_s args;

	result = cudaMalloc((void **)&d_data, dataSize*sizeof(int));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_data: %s\n", cudaGetErrorString(result));
	//result = cudaMalloc((void **)&d_stats, sizeof(Statistics));
	//if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_stats: %s\n", cudaGetErrorString(result));
	result = cudaMalloc((void **)&d_stats, txCount*sizeof(Statistics));
	if(result != cudaSuccess) fprintf(stderr, "Failed to allocate d_ratio: %s\n", cudaGetErrorString(result));
	
	uint64_t *sumNbAborts;
	uint64_t *sumNbCommits;

	CUDA_CHECK_ERROR(cudaMallocManaged(&sumNbCommits, sizeof(uint64_t)), "Could not alloc");
	CUDA_CHECK_ERROR(cudaMallocManaged(&sumNbAborts, sizeof(uint64_t)), "Could not alloc");

	*sumNbAborts = 0;
	*sumNbCommits = 0;

	CUDA_CPY_TO_DEV(d_data, h_data, dataSize*sizeof(int));
	CUDA_CPY_TO_DEV(d_stats, h_stats, txCount*sizeof(Statistics));

	char *gpu_txs;
    size_t tx_size = txCount * sizeof(YCSBTx);
    cudaMalloc(&gpu_txs, tx_size);
    cudaMemcpy(gpu_txs, txdata, tx_size, cudaMemcpyHostToDevice);

  	///////////////
	//kernel stuff
/*	totT_ms = parent_kernel(blockSize, txCount, batchSize, valid_txn_bitoffset,
								gpu_txs,
								dataSize, d_data,
								args,
								d_stats);

	__host__ float parent_kernel(uint block_size,
								int txcnt, int batch_size, int valid_txn_bitoffset,
								char* txs,
								uint dataSize, int* data, 
								PR_globalKernelArgs,
								Statistics* stats) {
*/
	float tKernel_ms = 0.0, totT_ms = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    std::vector<size_t> batches;
    std::vector<cudaStream_t> streams;
  	
  	for (size_t st = 0; st < txCount; st += batch_size)
    	batches.push_back(std::min<unsigned long>(batch_size, txCount - st));
    streams.resize(batches.size());

    int batch_st = 0;
    int batch_en = 0;

    for (int i = 0; i < batches.size(); i++)
    {
        size_t batched_txnum = batches[i];
        if (!batched_txnum)
            continue;
        batch_en = batch_st + batched_txnum;
        size_t thread_num = batched_txnum << valid_txn_bitoffset;

        dim3 num_blocks(thread_num / block_size + (thread_num % block_size == 0 ? 0 : 1), 1, 1);
        dim3 num_threads(thread_num > block_size ? block_size : thread_num, 1, 1);

        PR_blockNum = thread_num / block_size + (thread_num % block_size == 0 ? 0 : 1);
		PR_threadNum = thread_num > block_size ? block_size : thread_num;

        cudaEventRecord(start);
        PR_prepare_noCallback(&args);

		client_kernel<<<num_blocks, num_threads, 0, streams[i]>>>(
			dataSize, 
			d_data,
			gpu_txs,
			batch_st,
			batch_en,
			args.dev,
			d_stats);
        
        PR_postrun_noCallback(&args);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&tKernel_ms, start, stop);
        totT_ms += tKernel_ms;

        PR_reduceCommitAborts<<<PR_blockNum, PR_threadNum, 0, PR_streams[PR_currentStream]>>> (0, PR_currentStream, args.dev, sumNbCommits, sumNbAborts);

        batch_st = batch_en;
    }
    for (auto &stream : streams)
    	cudaStreamSynchronize(stream);

	
	//Copy metric data back to the host
	//cudaMemcpy(h_stats, d_stats, sizeof(Statistics), cudaMemcpyDeviceToHost);
  	cudaMemcpy(h_stats, d_stats, txCount*sizeof(Statistics), cudaMemcpyDeviceToHost);

  	getKernelOutput(h_stats, txCount, peak_clk, totT_ms, *sumNbCommits, *sumNbAborts, verbose);


	free(h_data);
	cudaFree(d_data);
	free(h_stats);
	cudaFree(d_stats);
	//cudaFree(d_times);
}

int main(int argc, char *argv[]) {

	const char APP_HELP[] = ""                
	  "argument order:                     \n"
	  "  1) input file 		               \n"
	  "  2) data size 					   \n"
	  "  3) nb txs 					       \n"
	  "  4) valid tx bit offset            \n"
	  "  5) block size                     \n"
	  "  6) batch size                     \n"
	  "  7) verbose		                   \n"
	"";
	const int NB_ARGS = 8;
	
	if (argc != NB_ARGS) {
		printf("%s\n", APP_HELP);
		exit(EXIT_SUCCESS);
	}

    int dataSize, txCount, valid_txn_bitoffset, blockSize, batchSize, verbose;
    sscanf(argv[2], "%d", &dataSize);
    sscanf(argv[3], "%d", &txCount);
    sscanf(argv[4], "%d", &valid_txn_bitoffset);
    sscanf(argv[5], "%d", &blockSize);
    sscanf(argv[6], "%d", &batchSize);
    sscanf(argv[7], "%d", &verbose);
    if (blockSize > 1024 || valid_txn_bitoffset > 5)
        return -1;

	YCSBTx *txdata = new YCSBTx[txCount];

	FILE *file = fopen(argv[1], "rb");
    fread(txdata, sizeof(YCSBTx), txCount, file);
    fclose(file);
	
	cudaSetDevice(0);
	for (int i = 0; i < 1; i++) {
		test_fine_grain_offloading(txdata, dataSize, txCount, valid_txn_bitoffset, blockSize, batchSize, verbose);
	}
	return 0;
}
