#ifndef _UNIFORMGRIDS_
#define _UNIFORMGRIDS_

#include "UtilsDefs.h"
//#include <queue>
const bool VALID = true;
const bool INVALID = false;

const int MAX_PER_GRID = 20;
using namespace std;

struct Grid
{
	int morton;
	int num;
	Point2D points[MAX_PER_GRID];
	bool valid[MAX_PER_GRID];
	int priority[MAX_PER_GRID];
	Grid(){
		morton = 0;
		num = 0;
	}
};
const double EPS = 0.000001;

//2D UniformGrids class
class UniformGrids
{
public:
	UniformGrids(){}
	~UniformGrids();
	//Construct empty UniformGrids based on dimension
	UniformGrids(int dimension);

	//create with width and height
	UniformGrids(int w, int h, BBox _bbox);

	inline unsigned int SpatialToIdx(Point2D point){return (unsigned int) (((int)((point.y - gridBbox.ymin)*height)) * width + (point.x - gridBbox.xmin) * width);}
	
	//insert a point with priority
	void insert(Point2D point, int priority = 0);

	//return false if there are some points within range.otherwise return true;
	bool dartSearch(Point2D point, double range);
	bool dartSearch(Point2D point, double range, bool& same);
	//push all points withing the ring (from inner range to outer range) to the buffer buf
	void ringSearch_buffer(Point2D point, double range_inner, double range_outer, vector<Point2D>& buf);	
	void dartSearch_buffer(Point2D point, double range, vector<Point2D>& buf);

	//push all conflict points to conflictbuffer, record their priority, grididx and in_grid idx;
	void dartSearch_buffer_pri(Point2D point, double range, vector<Point2D>& conflictBuffer,
		vector<int>& conflictPri, vector<int>& conflictGrididx, vector<int>& conflictIngrid_idx);
	
	//insert in the gap, for pivot point and its corresponding conflict points
	void insert_in_gap(Point2D pivotPoint, Point2D conflictPoint, double radius);

	//for each pivot_point, do process in the paper
	//attention: parameter is radius
	void process_pivot_point(Point2D& pivotPoint, int pri, double radius, int grid_idx, int ingrid_idx);
	
	//wrapper function, call this to eliminate all conflict points, and insert points for maximal coverage.
	//attention: parameter is radius
	void eliminate_for_maximal(double radius);

	//wrapper function, climinate all conflict points, insert points for maximal coverage.
	//This function will insert points on corners.
	void eliminate_for_maximal_corner(double radius);

	void process_pivot_point_corner(Point2D& pivotPoint, int pri, double radius, int grid_idx, int ingrid_idx);
	void insert_in_gap_corner(Point2D pivotPoint, Point2D conflictPoint, double radius);


	//wrapper function, this function eliminate all conflict points, insert points at once.
	void eliminate_for_maximal_batch(double radius);
	void process_pivot_point_batch(Point2D& pivotPoint, int pri, double radius, int grid_idx, int ingrid_idx);
	void insert_in_gap_batch(Point2D pivotPoint, vector<Point2D> conflictPoint, double radius);

	//attention: parameter is radius
	void test_maximal(double radius);

	int getCount(){int count = 0;
		for (int i = 0; i < dimension_edge*dimension_edge; i++){
			for (int j = 0; j < grids[i].num; j++){
				if (grids[i].valid[j])count++;
			}
		}return count;
	}
	void printToFile(string output_file_name)
	{
		ofstream fout_result(output_file_name);
		for (int i = 0; i <dimension_edge *dimension_edge; i++)
		{
			if (grids[i].num >= 1)
			{
				for (int j = 0; j < grids[i].num; j++)
				{
					if (grids[i].valid[j])
					{
						fout_result << grids[i].points[j].x << " " << grids[i].points[j].y << endl;

					}

				}
			}
		}
	}

	//print all grid information
	void print();
	Grid* grids;
	int dimension_edge;
	int width;
	int height;

	//bbox with actual coordinate
	BBox gridBbox;
	float w_f;
	float h_f;

	double itval;
	double itval_w;
	double itval_h;
};

#endif