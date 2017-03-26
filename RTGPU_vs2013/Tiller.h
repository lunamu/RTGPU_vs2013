#pragma once

#include "UtilsDefs.h"
#include "UniformGrids.h"
#include "KD_tree.h"

//This is a class in charge of tilling.
//Main task: input point pattern, tile point sampling, and 


class Tiller {

public:
	Tiller() {};
	Tiller(BBox bbox_, KDnode* kd_, float radius): bbox(bbox_),kd_tree(kd_) { }
	
	~Tiller() {};
	//void insert_in_gap(Point2D pivotPoint, Point2D conflictPoint, double radius);
	//void eliminate_for_maximal(double radius);
	//void process_pivot_point(Point2D& pivotPoint, int pri, double radius, int grid_idx, int ingrid_idx);
	void DivideConquerTiling(BBox bbox, double r, int axis, float ratio);
	void tilePoints(BBox bbox,Point2D offset, float ratio){
		traverse_and_classify(kd_tree,bbox, offset, ratio);
	}
	void traverse_and_classify(KDnode* root, BBox query, Point2D offset, float ratio);

	//int getGridCount() { return points_in_grid.getCount(); }

	/*void printToFile(std::string output_file_name)
	{
		points_in_grid.printToFile(output_file_name);
	}*/
	void printToFile(std::string output_file_name)
	{
		ofstream out_file(output_file_name);
		for (int i = 0; i < result.size(); i++)
		{
			out_file << result[i].x << " " << result[i].y << endl;
		}
	}

	BBox bbox;
	KDnode* kd_tree;
	//UniformGrids points_in_grid;
	vector<Point2D> result;
	vector<int> priority;
	//vector<Point2D> pattern;
};