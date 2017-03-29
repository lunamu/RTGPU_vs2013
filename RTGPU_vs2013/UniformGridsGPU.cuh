#pragma once

#include <vector>
#include "UtilsDefs.h"
#include "UtilKernels.cuh"
#include <cuda.h>
using namespace std;


struct by_hash {
	bool operator()(PointWithHash const &a, PointWithHash const &b) {
		return a.hash < b.hash;
	}
};


#define MAX_LOCAL_NUM 20
class UniformGridsGPU
{
public:
	UniformGridsGPU() {}
	UniformGridsGPU(vector<Point2D>& host_points, vector<int>& host_priority, int dim_);

	void gpu_eliminate_conflict_points(float range);

	void gpu_insert_in_gap(float range);
	//void gpu_gen_gap_points();
	//void gpu_eliminate_gap_points();

	//Input a point struct array, make it a grid. Also replace the grid in this class
	//this is not suitable for this class. But as a way to simplify the whole process.
	//everytime it get dev_compacted_points, sort it by hash value, generate new dev_idx array.
	void gpu_gridize();


	//Naming: don't name your parameters like functions. Important. Corret later.
	Point2D* dev_points;
	Point2D* dev_gap_points;

	int* dev_hash;
	int* dev_priority;

	unsigned int* dev_num_per_grid;
	
	//each store an index, indicate the index of a certain hash value in the point array.
	int* dev_idx;

	//mark array. if it is the start of a grid, mark its hash value. otherwise, mark -1.
	int* dev_mark;

	//valid array, mark if this point is still valid after elimination
	//already contained in point struct.
	bool* dev_valid;

	//if this point did a point elimination, set this to true
	bool* dev_primary;

	//dev_points_hash_priority;
	PointStruct* dev_php;

	Point2D* dev_compacted_points;

	int compacted_point_num;
	
	PointStruct* dev_compacted_php;
	int compacted_php_num;

	//Those points that actually generated the gap
	PointStruct* dev_gap_origin_php;
	int gap_origin_php_num;


	PointStruct* dev_gap_php;
	int* dev_gap_php_per_primary_num;

	//only store the number, as a test.
	int* dev_gap_php_num;

	//Some test cases
	float* dev_information_buffer;


	//Test buf
	//int* dev_

	float rad;
	BBox gridBbox;
	float w_f;
	float h_f;

	double itval;
	int dim;
	int width;
	int height;

	int point_num;
};