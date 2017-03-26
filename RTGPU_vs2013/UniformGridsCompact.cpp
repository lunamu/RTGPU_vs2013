#include "UniformGridsCompact.h"


UniformGridsCompact::UniformGridsCompact()
{
}

UniformGridsCompact::UniformGridsCompact(double radius, vector<Point2D> input_points)
{
	int base_dimension = 1.0 / radius;
	
	unsigned int r = 0; // r will be lg(v)

	while (base_dimension >>= 1) // unroll for more speed...
	{
		r++;
	}
	dim = 2 ^ (r + 1);
	width = dim;
	height = dim;
	

	
	for (Point2D p : input_points)
	{
		PointWithHash pwh; pwh.p = p; pwh.hash = SpatialToIdx(p);
		all_points.push_back(pwh);
	}

	//sort points array based, use all_hash as key.
	std::sort(all_points.begin(), all_points.end(), by_hash());
	idx.resize(dim * dim);
	for (int i = 0; i < idx.size(); i++) { idx[i] = -1; }
	for (int i = 0; i < all_points.size(); i++)
	{
		//unsigned int current_hash = all_points[i].hash;
		idx[all_points[i].hash] = i;
	}
}
