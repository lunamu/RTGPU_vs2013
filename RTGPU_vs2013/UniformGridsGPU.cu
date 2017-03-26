#include "UniformGridsGPU.cuh"


__device__ inline unsigned int SpatialToHash(Point2D point, BBox gridBbox, int height, int width) 
{ return (unsigned int)(((int)((point.y - gridBbox.ymin)*height)) * width + (point.x - gridBbox.xmin) * width); }

__constant__ double EPS = 0.000001;


__global__ void global_and(bool* A, bool* B, int num)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num)
	{
		A[idx] = A[idx] && B[idx];
	}
}

__global__ void cuda_generate_hash_value(int size, 
	PointStruct* dev_php, 
	Point2D* dev_points, 
	int* dev_priority,
	int* dev_hash,
	bool* dev_valid,
	int* dev_mark,
	BBox gridBbox,
	int height,
	int width)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		dev_php[idx].p = dev_points[idx];
		dev_php[idx].priority = dev_priority[idx];
		dev_php[idx].valid = true;
		int hash = SpatialToHash(dev_points[idx], gridBbox, height, width);
		dev_php[idx].hash = hash;
		dev_hash[idx] = hash;
		dev_valid[idx] = true;
		/*if (idx == 0)dev_mark[idx] = dev_hash[idx];
		else
		{
			if (dev_hash[idx] == dev_hash[idx - 1])dev_mark[idx] = -1;
			else dev_mark[idx] = dev_hash[idx];
		}*/
	}
}

__global__ void cuda_init_index(int size, int* dev_idx )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		dev_idx[idx] = -1;
	}
}


__global__ void cuda_generate_mark(int size, int* dev_hash, int* dev_mark)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		if (idx == 0)dev_mark[idx] = dev_hash[idx];
		else
		{
			if (dev_hash[idx] == dev_hash[idx - 1])dev_mark[idx] = -1;
			else dev_mark[idx] = dev_hash[idx];
		}
	}
}
__global__ void cuda_generate_index(int size, int* dev_mark, int* dev_idx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		if (dev_mark[idx] != -1)dev_idx[dev_mark[idx]] = idx;
	}
}

__device__ inline double devDistanceSquared(Point2D p1, Point2D p2)
{
	return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}
__device__ bool dart_search(Point2D query, float range,
							BBox gridBbox,
	int *dev_idx,
	int *dev_hash,
	int point_num,
	Point2D* dev_points,
	int width,
	int height)
{
	unsigned int x_start = max(0.0, (query.x - range - gridBbox.xmin)) * width;
	unsigned int x_end = min(gridBbox.xmax - EPS - gridBbox.xmin, (query.x + range - gridBbox.xmin)) * width;
	unsigned int y_start = max(0.0, (query.y - range - gridBbox.ymin)) * height;
	unsigned int y_end = min(gridBbox.ymax - EPS - gridBbox.ymin, (query.y + range - gridBbox.ymin)) * height;
	
	for (int i = x_start; i <= x_end; i++)
	{
		for (int j = y_start; j <= y_end; j++)
		{
			unsigned int idx = j * width + i;
			if (dev_idx[idx] == -1)
			{
				continue;
			}
			else
			{
				int grid_index = dev_idx[idx];
				int grid_hash = dev_hash[idx];
				int in_grid_idx = 0;
				while ((grid_index + in_grid_idx < point_num) && (dev_hash[grid_index + in_grid_idx] == grid_hash)) {

					double distSquare = devDistanceSquared(query, dev_points[grid_index + in_grid_idx]);
					if (distSquare == 0.0)continue;
					//if (distSquare < 1e-20)continue;
					else if (distSquare < range * range)
					{
						return false;
					}
					in_grid_idx++;
				}

			}
		}
	}
	return true;

}


__device__ void eliminate_conflict(int query_idx, float range, int* cp_index_buffer,
	BBox gridBbox,
	int *dev_idx,
	int *dev_hash,
	int point_num,
	Point2D* dev_points,
	int width,
	int height,
	int* dev_priority,
	bool* dev_valid,
	int* dev_information_buffer)//conflict points index buffer.
{


	int current_priority = dev_priority[query_idx];
	if (!dev_valid[query_idx])return;
	//if (current_priority == -1) return;
	int p_num = 0;
	Point2D query = dev_points[query_idx];
	unsigned int x_start = max(0.0, (query.x - range - gridBbox.xmin)) * width;
	unsigned int x_end = min(gridBbox.xmax - EPS - gridBbox.xmin, (query.x + range - gridBbox.xmin)) * width;
	unsigned int y_start = max(0.0, (query.y - range - gridBbox.ymin)) * height;
	unsigned int y_end = min(gridBbox.ymax - EPS - gridBbox.ymin, (query.y + range - gridBbox.ymin)) * height;
	//for (int i = x_start; i <= x_end; i++)
	//{
	//	for (int j = y_start; j <= y_end; j++)
	//	{
	//		unsigned int idx = j * width + i;
	//		if (dev_idx[idx] == -1)
	//		{
	//			continue;
	//		}
	//		else
	//		{
	//			int grid_index = dev_idx[idx];
	//			int grid_hash = dev_hash[grid_index];
	//			int in_grid_idx = 0;
	//			while ( (grid_index + in_grid_idx < point_num) && (dev_hash[grid_index + in_grid_idx] == grid_hash) ) 
	//			{
	//				double distSquare = devDistanceSquared(query, dev_points[grid_index + in_grid_idx]);
	//				if (distSquare < range * range)
	//				{
	//					if (dev_priority[grid_index + in_grid_idx] > current_priority)
	//					{
	//						eliminate_conflict(grid_index + in_grid_idx, range, cp_index_buffer, gridBbox, dev_idx, dev_hash, point_num, dev_points, width, height, dev_priority, dev_valid, dev_information_buffer);
	//						return;
	//					}
	//					else if (grid_index + in_grid_idx != query_idx)
	//					{
	//						//cp_index_buffer[p_num++] = grid_index + in_grid_idx;
	//						dev_valid[grid_index + in_grid_idx] = false;
	//					}
	//					
	//				}
	//				in_grid_idx++;
	//			}
	//		}
	//	}
	//}
	if (current_priority != -1)dev_valid[query_idx] = false;
	//dev_information_buffer[query_idx] = p_num;
	//if (p_num > 0)
	//{
	//	for (int i = 0; i < p_num; i++)
	//	{
	//		//eliminate the point;
	//		//if (dev_priority[cp_index_buffer[i]] < current_priority)
	//				dev_valid[cp_index_buffer[i]] = false; 
	//		
	//	}
	//}
}

__global__ void cuda_eliminate_conflict(float range, int size,
	BBox gridBbox,
	int *dev_idx,
	int *dev_hash,
	int point_num,
	Point2D* dev_points,
	int width,
	int height,
	int* dev_priority,
	bool* dev_valid,
	int* dev_information_buffer)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		//per thread storage
		dev_information_buffer[idx] = 0;
		int cp_index_buffer[MAX_LOCAL_NUM];
		eliminate_conflict(idx, range, cp_index_buffer,gridBbox, dev_idx, dev_hash, point_num, dev_points, width, height, dev_priority, dev_valid, dev_information_buffer);
	}
}

__device__ void eliminate_conflict_php(int query_idx, float range, int* cp_index_buffer,
	BBox gridBbox,
	PointStruct* dev_php,
	int *dev_idx,
	int point_num,
	bool* dev_valid,
	int width,
	int height,
	int* dev_information_buffer,bool* dev_primary)//conflict points index buffer.
{

	dev_primary[query_idx] = false;
	int current_priority = dev_php[query_idx].priority;
	if (!dev_valid[query_idx])return;
	if (current_priority == -1) return;
	int p_num = 0;
	Point2D query = dev_php[query_idx].p;
	unsigned int x_start = max(0.0, (query.x - range - gridBbox.xmin)) * width;
	unsigned int x_end = min(gridBbox.xmax - EPS - gridBbox.xmin, (query.x + range - gridBbox.xmin)) * width;
	unsigned int y_start = max(0.0, (query.y - range - gridBbox.ymin)) * height;
	unsigned int y_end = min(gridBbox.ymax - EPS - gridBbox.ymin, (query.y + range - gridBbox.ymin)) * height;
	for (int i = x_start; i <= x_end; i++)
	{
		for (int j = y_start; j <= y_end; j++)
		{
			unsigned int idx = j * width + i;
			if (dev_idx[idx] == -1)
			{
				continue;
			}
			else
			{
				int grid_index = dev_idx[idx];
				int grid_hash = dev_php[grid_index].hash;
				int in_grid_idx = 0;
				while ( (grid_index + in_grid_idx < point_num) && (dev_php[grid_index + in_grid_idx].hash == grid_hash) ) 
				{
					double distSquare = devDistanceSquared(query, dev_php[grid_index + in_grid_idx].p);
					if (distSquare < range * range)
					{
						if (dev_php[grid_index + in_grid_idx].priority > current_priority)
						{
							eliminate_conflict_php(grid_index + in_grid_idx, range, cp_index_buffer, gridBbox, dev_php, dev_idx, point_num, dev_valid, width, height, dev_information_buffer, dev_primary);
							return;
						}
						else if (grid_index + in_grid_idx != query_idx)
						{
							dev_primary[query_idx] = true;
							//cp_index_buffer[p_num++] = grid_index + in_grid_idx;
							dev_valid[grid_index + in_grid_idx] = false;
						}
						
					}
					in_grid_idx++;
				}
			}
		}
	}
	//if (current_priority != -1)dev_valid[query_idx] = false;
	//dev_information_buffer[query_idx] = p_num;
	//if (p_num > 0)
	//{
	//	for (int i = 0; i < p_num; i++)
	//	{
	//		//eliminate the point;
	//		//if (dev_priority[cp_index_buffer[i]] < current_priority)
	//				dev_valid[cp_index_buffer[i]] = false; 
	//		
	//	}
	//}
}
__global__ void cuda_eliminate_conflict_php(float range, int size,
	BBox gridBbox,
	int *dev_idx,
	PointStruct* dev_php,
	int point_num,
	bool* dev_valid,
	int width,
	int height,
	int* dev_information_buffer, bool* dev_primary)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		//per thread storage
		dev_information_buffer[idx] = 0;
		int cp_index_buffer[MAX_LOCAL_NUM];
		eliminate_conflict_php(idx, range, cp_index_buffer, gridBbox, dev_php,dev_idx, point_num, dev_valid,width, height, dev_information_buffer, dev_primary);
	}
}
//__global__ void cuda_insert_in_gap(flaot range, int size, )

UniformGridsGPU::UniformGridsGPU(vector<Point2D>& host_points, vector<int>& host_priority, int dim_)
{
	//Set some class parameters
	dim = dim_;
	gridBbox = gUnitGrid;
	w_f = gridBbox.xmax - gridBbox.xmin;
	h_f = gridBbox.ymax - gridBbox.ymin;
	itval = 1.0 / dim;
	width = dim_;
	height = dim_;

	point_num = host_points.size();
	//allocate GPU memory;
	gpuErrchk(cudaMalloc((void**)&dev_points, sizeof(Point2D) * host_points.size()));
	gpuErrchk(cudaMalloc((void**)&dev_idx, sizeof(int) * width * height));
	gpuErrchk(cudaMalloc((void**)&dev_hash, sizeof(int) * host_points.size()));
	gpuErrchk(cudaMalloc((void**)&dev_priority, sizeof(int) * host_priority.size()));
	gpuErrchk(cudaMalloc((void**)&dev_php, sizeof(PointStruct) * host_points.size()));
	gpuErrchk(cudaMalloc((void**)&dev_valid, sizeof(bool) * point_num));
	gpuErrchk(cudaMalloc((void**)&dev_primary, sizeof(bool) * point_num));
	gpuErrchk(cudaMalloc((void**)&dev_mark, sizeof(int) * point_num));
	gpuErrchk(cudaMalloc((void**)&dev_compacted_points, sizeof(Point2D) * point_num));
	gpuErrchk(cudaMalloc((void**)&dev_compacted_php, sizeof(PointStruct) * point_num));
	gpuErrchk(cudaMalloc((void**)&dev_gap_origin_php, sizeof(PointStruct) * point_num))
	gpuErrchk(cudaMalloc((void**)&dev_information_buffer, sizeof(int) * point_num));


	//copy points and priority from host
	gpuErrchk(cudaMemcpy(dev_points, &host_points[0], sizeof(Point2D) * point_num, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_priority, &host_priority[0], sizeof(int) * point_num, cudaMemcpyHostToDevice))


	//Modify: This might not be correct

	//Generate hash for all points
	int threadsPerBlock = 256;
	int numBlocks = (point_num + threadsPerBlock - 1) / threadsPerBlock;
	cuda_generate_hash_value << <numBlocks, threadsPerBlock >> > (point_num, dev_php,
		dev_points,
		dev_priority,
		dev_hash,
		dev_valid,
		dev_mark,
		gridBbox,
		height,
		width);

	//backup hashvalue array
	int* dev_hash_backup;
	gpuErrchk(cudaMalloc((void**)&dev_hash_backup, sizeof(int) * host_points.size()));
	gpuErrchk(cudaMemcpy(dev_hash_backup, dev_hash, sizeof(int) * point_num, cudaMemcpyDeviceToDevice));

	thrust::device_ptr<int> dev_hash_pointer(dev_hash);
	thrust::device_ptr<int> dev_hash_backup_pointer(dev_hash_backup);
	//thrust::device_ptr<PointStruct> dev_php_pointer(dev_php);
	thrust::sort_by_key(thrust::device, dev_hash_pointer, dev_hash_pointer + point_num, dev_points);
	thrust::sort_by_key(thrust::device, dev_hash_backup_pointer, dev_hash_backup_pointer + point_num, dev_php);

	//viewGPUArray<int>(dev_hash, point_num, "dev_hash");

	//Generate idx
	cuda_init_index << <numBlocks, threadsPerBlock >> >(point_num, dev_idx);
	gpuErrchk(cudaGetLastError());
	cuda_generate_mark << <numBlocks, threadsPerBlock >> >(point_num, dev_hash, dev_mark);
	gpuErrchk(cudaGetLastError());
	cuda_generate_index << <numBlocks, threadsPerBlock >> >(point_num, dev_mark, dev_idx);
	gpuErrchk(cudaGetLastError());
	viewGPUArray<int>(dev_idx, dim * dim, "dev_idx");
	viewGPUArray<int>(dev_hash, point_num, "dev_hash");
	viewGPUPoint2D(dev_points, point_num, "dev_points");
	viewGPUArray<int>(dev_priority, point_num, "dev_priority");
}
struct is_true
{
	__host__ __device__
	bool operator()(const bool x)
	{
		return (x == true);
	}
};

void UniformGridsGPU::gpu_eliminate_conflict_points(float range)
{
	int threadsPerBlock = 256;
	int numBlocks = (point_num + threadsPerBlock - 1) / threadsPerBlock;
	/*cuda_eliminate_conflict << <numBlocks, threadsPerBlock >> >  (range, 
		point_num, 
		gridBbox, 
		dev_idx, 
		dev_hash,
		point_num, 
		dev_points, 
		width, height, 
		dev_priority, 
		dev_valid,
		dev_information_buffer
		)*/;

	cuda_eliminate_conflict_php << <numBlocks, threadsPerBlock >> >  (range,
		point_num,
		gridBbox,
		dev_idx, dev_php, point_num, dev_valid, width, height, dev_information_buffer,dev_primary);
	gpuErrchk(cudaGetLastError());
	//thrust::device_ptr<Point2D> dev_points_pointer(dev_points);
	//thrust::device_ptr<bool> dev_valid_pointer(dev_valid);
	//thrust::device_ptr<Point2D> dev_compacted_points_pointer(dev_compacted_points);
	//thrust::copy_if(thrust::device, dev_points_pointer, dev_points_pointer + point_num, dev_valid_pointer, dev_compacted_points_pointer, is_true());
	//Point2D* last = thrust::copy_if(thrust::device, dev_points, dev_points + point_num, dev_valid, dev_compacted_points, is_true());
	//compacted_point_num = last - dev_compacted_points;
	//viewGPUArray<bool>(dev_valid, point_num, "dev_valid");
	PointStruct* last = thrust::copy_if(thrust::device, dev_php, dev_php + point_num, dev_valid, dev_compacted_php , is_true());
	compacted_php_num = last - dev_compacted_php;

	global_and << <numBlocks, threadsPerBlock >> >(dev_primary, dev_valid, point_num);
	last = thrust::copy_if(thrust::device, dev_php, dev_php + point_num, dev_primary, dev_gap_origin_php, is_true());
	gap_origin_php_num = last - dev_gap_origin_php;
	viewGPUArray<int>(dev_information_buffer, point_num, "dev_information_buffer");
	viewGPUPoint2D(dev_gap_origin_php, gap_origin_php_num, "dev_gap_origin");
}

void UniformGridsGPU::gpu_insert_in_gap(float range)
{
	int threadsPerBlock = 256;
	int numBlocks = (point_num + threadsPerBlock - 1) / threadsPerBlock;

	//First, let's do a mega kernel.
	//This kernel will take every points that ever did an elimination in the point set as input.
	//

	//generate point set to do the gap checking.
}