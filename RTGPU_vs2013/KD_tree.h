#ifndef __KD_TREE__
#define __KD_TREE__

#include "UtilsDefs.h"
#include "UniformGrids.h"

using namespace std;

class KDnode
{

public:
	int num;
	bool leaf;
	Point2D* pts;
	char split_axis;
	float split_position;
	BBox cell_bbox;
	KDnode* left;
	KDnode* right;
};


KDnode* build(Point2D* points, int num, int depth);
void traverse(KDnode* root, vector<Point2D>& result, BBox query);
//void traverse_and_classify(KDnode*root, vector<Point2D>& result, vector<int>& priority, BBox query, Point2D offset,float ratio);
void traverse_with_priority(KDnode* root, vector<Point2D>& result, vector<int>& priority, BBox query, int pri);
#endif