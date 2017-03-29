#include "UniformGridsGPU.cuh"


__constant__ double EPS = 0.000001;


__device__ inline unsigned int SpatialToHash(Point2D point, BBox gridBbox, int height, int width)//potential bug 
{
	return (unsigned int)(((int)((point.y - EPS - gridBbox.ymin)*height)) * width + (point.x - gridBbox.xmin) * width);
}

__global__ void global_and(bool* A, bool* B, int num)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num)
	{
		A[idx] = A[idx] && B[idx];
	}
}

__global__ void global_and(int* A, bool* B, int num)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num)
	{
		A[idx] = (bool)(A[idx]) && B[idx];
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
	int point_num,
	PointStruct* dev_php,
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
				int grid_hash = dev_php[grid_index].hash;
				int in_grid_idx = 0;
				while ((grid_index + in_grid_idx < point_num) && (dev_php[grid_index + in_grid_idx].hash == grid_hash)) {
					if (dev_php[grid_index + in_grid_idx].valid == false){
						in_grid_idx++;
						continue;
					}
					double distSquare = devDistanceSquared(query, dev_php[grid_index + in_grid_idx].p);
					if (distSquare == 0.0){
						in_grid_idx++;
						continue;
					}
					
					if (distSquare < range * range)
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
	float* dev_information_buffer)//conflict points index buffer.
{


	int current_priority = dev_priority[query_idx];
	if (!dev_valid[query_idx])return;
	//if (current_priority == -1) return;
	//int p_num = 0;
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
	float* dev_information_buffer)
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

__device__ void eliminate_conflict_php(int query_idx, float range,
	BBox gridBbox,
	PointStruct* dev_php,
	int *dev_idx,
	int point_num,
	bool* dev_valid,
	int width,
	int height,
	float* dev_information_buffer,bool* dev_primary)//conflict points index buffer.
{

	dev_primary[query_idx] = false;
	int current_priority = dev_php[query_idx].priority;
	if (!dev_valid[query_idx])return;
	if (current_priority == -1) return;
	//int p_num = 0;
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
							eliminate_conflict_php(grid_index + in_grid_idx, range, gridBbox, dev_php, dev_idx, point_num, dev_valid, width, height, dev_information_buffer, dev_primary);
							return;
						}
						else if (grid_index + in_grid_idx != query_idx)
						{
							dev_primary[query_idx] = true;
							//cp_index_buffer[p_num++] = grid_index + in_grid_idx;
							dev_valid[grid_index + in_grid_idx] = false;
							dev_php[grid_index + in_grid_idx].valid = false;
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
	float* dev_information_buffer, bool* dev_primary)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		//per thread storage
		dev_information_buffer[idx] = 0;
	
		
		eliminate_conflict_php(idx, range, gridBbox, dev_php,dev_idx, point_num, dev_valid,width, height, dev_information_buffer, dev_primary);
	}
}

__device__ inline bool Circumcenter(Point2D p0, Point2D p1, Point2D p2, Point2D& center, double& ra2)
{
	double dA, dB, dC, aux1, aux2, div;

	dA = p0.x * p0.x + p0.y * p0.y;
	dB = p1.x * p1.x + p1.y * p1.y;
	dC = p2.x * p2.x + p2.y * p2.y;

	aux1 = (dA*(p2.y - p1.y) + dB*(p0.y - p2.y) + dC*(p1.y - p0.y));
	aux2 = -(dA*(p2.x - p1.x) + dB*(p0.x - p2.x) + dC*(p1.x - p0.x));
	div = (2 * (p0.x*(p2.y - p1.y) + p1.x*(p0.y - p2.y) + p2.x*(p1.y - p0.y)));

	if (div == 0){
		return false;
	}

	center.x = aux1 / div;
	center.y = aux2 / div;
	ra2 = ((center.x - p0.x)*(center.x - p0.x) + (center.y - p0.y)*(center.y - p0.y));

	return true;
}

__global__ void cuda_insert_in_gap(float range, int size, BBox gridBbox,
	int *dev_idx, PointStruct* dev_gap_origin_php, PointStruct* dev_php, int point_num, int gap_origin_php_num, int* dev_gap_php_num, int width,
	int height, bool* dev_valid, Point2D* dev_gap_points, int* dev_global_gap_points_idx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < gap_origin_php_num)
	{
		Point2D pivot = dev_gap_origin_php[idx].p;
		unsigned int x_start = max(0.0, (pivot.x - 2 * range - gridBbox.xmin)) * width;
		unsigned int x_end = min(gridBbox.xmax - EPS - gridBbox.xmin, (pivot.x + 2 * range - gridBbox.xmin)) * width;
		unsigned int y_start = max(0.0, (pivot.y - 2 * range - gridBbox.ymin)) * height;
		unsigned int y_end = min(gridBbox.ymax - EPS - gridBbox.ymin, (pivot.y + 2 * range - gridBbox.ymin)) * height;
		Point2D test_buf[MAX_LOCAL_NUM];
		int test_point_num = 0;
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
					while ((grid_index + in_grid_idx < point_num) && (dev_php[grid_index + in_grid_idx].hash == grid_hash))
					{
						if (dev_valid[grid_index + in_grid_idx] == false){
							in_grid_idx++;
							continue;
						}
						Point2D test_p = dev_php[grid_index + in_grid_idx].p;
						float distSquare = devDistanceSquared(pivot, test_p);
						if (distSquare < 4 * range * range)
						{
							if (distSquare != 0)
							{
								test_buf[test_point_num++] = test_p;
							}

						}
						in_grid_idx++;
					}
				}
			}
		}

		int count_result = 0;
		//First, only pivot points and others.
		Point2D gap_buf[MAX_LOCAL_NUM];
		for (int i = 0; i < test_point_num; i++)
		{
			for (int j = i + 1; j < test_point_num; j++)
			{
				if (i == j) continue;
				Point2D center;
				double cir_r2;
				Circumcenter(pivot, test_buf[i], test_buf[j], center, cir_r2);
				if ((center.x >= gridBbox.xmax) || (center.x <= gridBbox.xmin) || (center.y >= gridBbox.ymax) || (center.y <= gridBbox.ymin))continue;
				if (dart_search(center, range, gridBbox, dev_idx, point_num, dev_php, width, height))
				{
					bool gap_valid = true;
					for (int ct = 0; ct < count_result; ct++)
					{
						if (devDistanceSquared(center, gap_buf[ct]) < range * range){
							gap_valid = false; break;
						};
					}
					if (gap_valid) gap_buf[count_result++] = center;
				}
			}
		}
		for (int i = 0; i < count_result; i++)
		{
			dev_gap_points[atomicAdd(dev_global_gap_points_idx, 1)] = gap_buf[i];
		}
	}
}

__global__ void cuda_test_conflict(float range,
	BBox gridBbox,
	PointStruct* dev_php,
	int *dev_idx,
	int point_num,
	bool* dev_valid,
	int width,
	int height,
	float* dev_information_buffer)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < point_num)
	{
		if (!dev_valid[idx])
		{
			dev_information_buffer[idx] = 1000;
			return;
		}
		Point2D current_point = dev_php[idx].p;
		float dist = 500000.0;
		unsigned int x_start = max(0.0, (current_point.x - range - gridBbox.xmin)) * width;
		unsigned int x_end = min(gridBbox.xmax - EPS - gridBbox.xmin, (current_point.x + range - gridBbox.xmin)) * width;
		unsigned int y_start = max(0.0, (current_point.y - range - gridBbox.ymin)) * height;
		unsigned int y_end = min(gridBbox.ymax - EPS - gridBbox.ymin, (current_point.y + range - gridBbox.ymin)) * height;
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
					while ((grid_index + in_grid_idx < point_num) && (dev_php[grid_index + in_grid_idx].hash == grid_hash)) {
						if (dev_php[grid_index + in_grid_idx].valid == false){
							in_grid_idx++;
							continue;
						}
						double distSquare = devDistanceSquared(current_point, dev_php[grid_index + in_grid_idx].p);
						if (distSquare == 0.0){
							in_grid_idx++;
							continue;
						}						
						float sq = sqrtf(distSquare);
						if (sq < dist)dist = sq;					
						in_grid_idx++;
					}

				}
			}
		}
		dev_information_buffer[idx] = dist;
		
	}
}

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
	gpuErrchk(cudaMalloc((void**)&dev_gap_origin_php, sizeof(PointStruct) * point_num));
	gpuErrchk(cudaMalloc((void**)&dev_information_buffer, sizeof(float) * point_num));
	gpuErrchk(cudaMalloc((void**)&dev_gap_points, sizeof(Point2D) * point_num * 10));


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
	viewGPUArray<int>(dev_idx, dim * dim, "dev_idx_ori");
	viewGPUArray<int>(dev_hash, point_num, "dev_hash");
	viewGPUPoint2D(dev_points, point_num, "dev_points");
	viewGPUArray<int>(dev_priority, point_num, "dev_priority");

}
struct is_true
{
	__host__ __device__
	bool operator()(const bool x)
	{
		return (x != 0 );
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
		)*/
	
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
	viewGPUPoint2D(dev_php, point_num, "php");
	
	compacted_php_num = last - dev_compacted_php;

	cuda_test_conflict << <numBlocks, threadsPerBlock >> >(range, gridBbox, dev_php, dev_idx, point_num, dev_valid, width, height, dev_information_buffer);
	
	viewGPUArray<float>(dev_information_buffer, point_num, "dev_information_buffer");


	//global_and << <numBlocks, threadsPerBlock >> >(dev_primary, dev_valid, point_num);
	//global_and << <numBlocks, threadsPerBlock >> >(dev_information_buffer, dev_valid, point_num);
	
	last = thrust::copy_if(thrust::device, dev_php, dev_php + point_num, dev_information_buffer, dev_gap_origin_php, is_true());
	gap_origin_php_num = last - dev_gap_origin_php;
	
	//viewGPUPoint2D(dev_gap_origin_php, gap_origin_php_num, "dev_gap_origin");
	viewGPUPoint2D(dev_gap_origin_php, gap_origin_php_num, "dev_conflict_point");

	
	gpuErrchk(cudaMalloc((void**)&dev_gap_php_num, sizeof(int)*gap_origin_php_num));
}

void UniformGridsGPU::gpu_insert_in_gap(float range)
{
	int threadsPerBlock = 256;
	int numBlocks = (point_num + threadsPerBlock - 1) / threadsPerBlock;

	//First, let's do a mega kernel.
	//This kernel will take every points that ever did an elimination in the point set as input.
	//

	//generate point set to do the gap checking.
	
	int host_global_counter = 0;
	int* dev_global_counter;
	cudaMalloc((void**)&dev_global_counter, sizeof(int) * 1);
	cudaMemcpy(dev_global_counter, &host_global_counter, sizeof(int) * 1, cudaMemcpyHostToDevice);

	viewGPUArray<int>(dev_idx, dim * dim, "dev_idx");

	cuda_insert_in_gap << <numBlocks, threadsPerBlock >> >(range, gap_origin_php_num, gridBbox,
		dev_idx, dev_gap_origin_php, dev_php, point_num, gap_origin_php_num, dev_gap_php_num, width,
		height, dev_valid, dev_gap_points, dev_global_counter);

	cudaMemcpy(&host_global_counter, dev_global_counter, sizeof(int) * 1, cudaMemcpyDeviceToHost);

	viewGPUArray<int>(dev_gap_php_num, gap_origin_php_num, "dev_gap_php_per_primary");
	viewGPUPoint2D(dev_gap_points, host_global_counter, "dev_gap_points");

}