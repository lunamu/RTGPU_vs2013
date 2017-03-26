#pragma once

#include "UtilsDefs.h"
#include "UniformGrids.h"
#include "KD_tree.h"



struct by_hash {
	bool operator()(PointWithHash const &a, PointWithHash const &b) {
		return a.hash < b.hash;
	}
};

class UniformGridsCompact
{
public:
	UniformGridsCompact();
	UniformGridsCompact(double radius,vector<Point2D> input_points);
	~UniformGridsCompact() {};

	inline unsigned int SpatialToIdx(Point2D point) { return (unsigned int)(((int)((point.y - gridBbox.ymin)*height)) * width + (point.x - gridBbox.xmin) * width); }

	

	//this vector stores all points;
	//use a very simple hash: hash value = dim * y_idx + x_idx;
	//TODO: Try morton hash value;

	vector<PointWithHash> all_points;

	
	//index of this vector is morton, value of this vector is the index in all_points.
	vector<int> idx;

	//bbox with actual coordinate
	BBox gridBbox;
	float w_f;
	float h_f;

	int dim;
	int width;
	int height;
};