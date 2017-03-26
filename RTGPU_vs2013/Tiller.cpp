#include "tiller.h"

inline int dimensionCheck(BBox bbox, double r)//consider leave a minimum leaf node.
{
	int x = 0; int y = 0;
	if (bbox.xmax - bbox.xmin > LEAFLEN)
	{
		x = 1;
	}
	if (bbox.ymax - bbox.ymin > LEAFLEN)
	{
		y = 1;
	}
	if (x == 1 && y == 1)return ALL;
	else if (x == 1)return XAXIS;
	else if (y == 1) return YAXIS;
	else return 9;
}

//
//void Tiller::insert_in_gap(Point2D pivotPoint, Point2D conflictPoint, double radius)
//{
//	vector<Point2D> pointTestBuffer;
//	points_in_grid.ringSearch_buffer(conflictPoint, 2 * radius, 4 * radius, pointTestBuffer);
//	
//	for (auto first_point = pointTestBuffer.begin(); first_point != pointTestBuffer.end(); first_point++)
//	{
//		for (auto second_point = first_point + 1; second_point != pointTestBuffer.end(); second_point++)
//		{
//			if (first_point == second_point) continue;
//			auto fp = *first_point;
//			auto sp = *second_point;
//			Point2D center;
//			double cir_r2;
//			Circumcenter(pivotPoint, fp, sp, center, cir_r2);
//			//if (cir_r2 > 16 * radius * radius)continue;
//			if (center.x > points_in_grid.gridBbox.xmax || center.x < points_in_grid.gridBbox.xmin || center.y > points_in_grid.gridBbox.ymax || center.y < points_in_grid.gridBbox.ymin)continue;
//			else if (points_in_grid.dartSearch(center, 2 * radius))
//			{
//				points_in_grid.insert(center);
//			}
//
//		}
//	}
//
//}
//
//void Tiller::process_pivot_point(Point2D& pivotPoint, int pri, double radius, int grid_idx, int ingrid_idx)
//{
//	vector<Point2D> conflictBuffer;
//	vector<int> conflictPri;
//	vector<int> conflictGrididx;
//	vector<int> conflictIngrid_idx;
//	points_in_grid.dartSearch_buffer_pri(pivotPoint, 2 * radius, conflictBuffer, conflictPri, conflictGrididx, conflictIngrid_idx);
//	for (int i = 0; i < conflictBuffer.size(); i++)
//	{
//		if (conflictPri[i] > pri)
//		{
//			process_pivot_point(conflictBuffer[i], conflictPri[i], radius, conflictGrididx[i], conflictIngrid_idx[i]);
//			return;
//		}
//	}
//	//cout << "Pivot point: " << pivotPoint.x << " " << pivotPoint.y << endl;
//	//cout << "Conflict point: " << endl;
//	/*for (int i = 0; i < conflictBuffer.size(); i++)
//	{
//	cout << conflictBuffer[i].x << " " << conflictBuffer[i].y << endl;
//	}*/
//	for (int i = 0; i < conflictBuffer.size(); i++)
//	{
//		int gidx = conflictGrididx[i];
//		int iidx = conflictIngrid_idx[i];
//		points_in_grid.grids[gidx].valid[iidx] = INVALID;	//eliminate;
//		insert_in_gap(pivotPoint, conflictBuffer[i], radius);
//	}
//}
//
//void Tiller::eliminate_for_maximal(double radius)
//{
//	for (int grid_idx = 0; grid_idx < points_in_grid.width * points_in_grid.height; grid_idx++)
//	{
//		if (points_in_grid.grids[grid_idx].num > 0)
//		{
//			for (int in_idx = 0; in_idx < points_in_grid.grids[grid_idx].num; in_idx++)//iteration of all points
//			{
//				Point2D& cur_point = points_in_grid.grids[grid_idx].points[in_idx];
//				int cur_pri = points_in_grid.grids[grid_idx].priority[in_idx];
//
//				process_pivot_point( cur_point, cur_pri, radius, grid_idx, in_idx);
//			}
//		}
//	}
//}


void Tiller::traverse_and_classify(KDnode*root, BBox query, Point2D offset, float ratio)
{
	if (withinBox(query, root->cell_bbox))
	{
		for (int i = 0; i < root->num; i++)
		{
			Point2D tmp_p = root->pts[i];
			//the query is the ratio box, map point in ratio box back.
			tmp_p.x -= query.xmin;
			tmp_p.y -= query.ymin;
			tmp_p.x /= ratio;
			tmp_p.y /= ratio;
			tmp_p.x += offset.x;
			tmp_p.y += offset.y;


			//if (within(primary_right, tmp_p) || within(primary_down, tmp_p))result.push_back(tmp_p);
			if (!within(ide, tmp_p))continue;

			result.push_back(tmp_p);
			if (within(secondary_left, tmp_p) || within(secondary_up, tmp_p)) {
				priority.push_back(0);
			}
			else if (within(primary_right, tmp_p) || within(primary_down, tmp_p))
			{
				priority.push_back(2);
			}
			else if (within(mid_NE, tmp_p) || within(mid_SW, tmp_p))
			{
				priority.push_back(1);
			}
			else
			{
				priority.push_back(-1);
			}
		}
	}
	else if (root->leaf == true)
	{
		for (int i = 0; i < root->num; i++)
		{
			if (within(query, root->pts[i]))
			{
				Point2D tmp_p = root->pts[i];
				tmp_p.x -= query.xmin;
				tmp_p.y -= query.ymin;
				tmp_p.x /= ratio;
				tmp_p.y /= ratio;
				tmp_p.x += offset.x;
				tmp_p.y += offset.y;

				if (!within(ide, tmp_p))continue;

				result.push_back(tmp_p);
				if (within(secondary_left, tmp_p) || within(secondary_up, tmp_p)) {
					priority.push_back(0);
				}
				else if (within(primary_right, tmp_p) || within(primary_down, tmp_p))
				{
					priority.push_back(2);
				}
				else if (within(mid_NE, tmp_p) || within(mid_SW, tmp_p))
				{
					priority.push_back(1);
				}
				else
				{
					priority.push_back(-1);
				}
			}
		}
	}
	else
	{
		if (root->split_axis == X_AXIS)
		{
			if (root->split_position >= query.xmin && root->split_position <= query.xmax)
			{
				traverse_and_classify(root->left,  query, offset, ratio);
				traverse_and_classify(root->right,  query, offset, ratio);
			}
			else if (query.xmax < root->split_position)
			{
				traverse_and_classify(root->left, query, offset, ratio);
			}
			else
			{
				traverse_and_classify(root->right,  query, offset, ratio);
			}
		}
		else
		{
			if (root->split_position >= query.ymin && root->split_position <= query.ymax)
			{
				traverse_and_classify(root->left, query, offset, ratio);
				traverse_and_classify(root->right,  query, offset, ratio);
			}
			else if (query.ymax < root->split_position)
			{
				traverse_and_classify(root->left, query, offset, ratio);
			}
			else
			{
				traverse_and_classify(root->right,  query, offset, ratio);
			}
		}
	}
}


void Tiller::DivideConquerTiling(BBox bbox, double r, int axis, float ratio)
{

	if (dimensionCheck(bbox, r) == ALL || dimensionCheck(bbox, r) == axis)
	{
		BBox reduced_bbox_left;
		BBox reduced_bbox_right;

		if (axis == XAXIS)
		{
			//Normal box
			reduced_bbox_left.xmin = bbox.xmin;
			reduced_bbox_left.xmax = IntervalNormal(bbox.xmin + 2 * r, bbox.xmax - 4 * r);
			//reduced_bbox_left.xmax = IntervalUniform(bbox.xmin + 2 * r, bbox.xmax - 4 * r);
			reduced_bbox_left.ymin = bbox.ymin; reduced_bbox_left.ymax = bbox.ymax;


			//!Modified for test
			//reduced_bbox_right.xmin = reduced_bbox_left.xmax + 2 * r;
			reduced_bbox_right.xmin = reduced_bbox_left.xmax + 2 * r;
			reduced_bbox_right.xmax = bbox.xmax;
			reduced_bbox_right.ymin = bbox.ymin; reduced_bbox_right.ymax = bbox.ymax;



			//blockingBox.push_back(BBox(reduced_bbox_left.xmax, bbox.ymin, 2 * r, bbox.ymax - bbox.ymin));
		}
		else
		{
			reduced_bbox_left.ymin = bbox.ymin;
			//reduced_bbox_left.ymax = IntervalUniform(bbox.ymin + 2 * r, bbox.ymax - 4 * r);
			reduced_bbox_left.ymax = IntervalNormal(bbox.ymin + 2 * r, bbox.ymax - 4 * r);
			reduced_bbox_left.xmin = bbox.xmin; reduced_bbox_left.xmax = bbox.xmax;

			//!Modified for test
			//reduced_bbox_right.ymin = reduced_bbox_left.ymax + 2 * r;
			reduced_bbox_right.ymin = reduced_bbox_left.ymax + 2 * r;
			reduced_bbox_right.ymax = bbox.ymax;
			reduced_bbox_right.xmin = bbox.xmin; reduced_bbox_right.xmax = bbox.xmax;


			//blockingBox.push_back(BBox(bbox.xmin, reduced_bbox_left.ymax, bbox.xmax - bbox.xmin, 2 * r));
		}

		//testing code
		//blockingBox

		DivideConquerTiling(reduced_bbox_left, r, -axis, ratio);
		DivideConquerTiling(reduced_bbox_right,  r, -axis, ratio);

	}
	else if (dimensionCheck(bbox, r) == -axis)
	{
		DivideConquerTiling(bbox, r, -axis, ratio);
	}
	else
	{
		double area = (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin);
		//genPoints(points, bbox, area, r);

		//rest is to use tiling
		double len = bbox.xmax - bbox.xmin + 4 * r;
		double height = bbox.ymax - bbox.ymin + 4 * r;
		double new_xmin = IntervalUniform(0.0, 1.0 - ratio * len);
		double new_ymin = IntervalUniform(0.0, 1.0 - ratio * height);
		BBox ratioBbox(new_xmin, new_xmin + ratio * len, new_ymin, new_ymin + ratio * height);

		//Primary & Secondary box
		primary_right = BBox(bbox.xmax, bbox.xmax + 2 * r, bbox.ymin - 2 * r, bbox.ymax - 2 * r);
		primary_down = BBox(bbox.xmin + 2 * r, bbox.xmax, bbox.ymin - 2 * r, bbox.ymin);
		secondary_left = BBox(bbox.xmin - 2 * r, bbox.xmin + 2 * r, bbox.ymin, bbox.ymax + 2 * r);
		secondary_up = BBox(bbox.xmin + 2 * r, bbox.xmax, bbox.ymax - 2 * r, bbox.ymax + 2 * r);
		mid_NE = BBox(bbox.xmax, bbox.xmax + 2 * r, bbox.ymax - 2 * r, bbox.ymax + 2 * r);
		mid_SW = BBox(bbox.xmin - 2 * r, bbox.xmin + 2 * r, bbox.ymin - 2 * r, bbox.ymin);
		tilePoints(ratioBbox, Point2D(bbox.xmin - 2 * r, bbox.ymin - 2 * r), ratio);


		//rest is to use regular tiling
		//double len = bbox.xmax - bbox.xmin;
		//double height = bbox.ymax - bbox.ymin;
		//double new_xmin = 0.0;// IntervalUniform(0.0, 1.0 - ratio * len);
		//double new_ymin = 0.0;
		//BBox ratioBbox(new_xmin, new_xmin + ratio * len, new_ymin, new_ymin + ratio * height);
		//tilePoints(ratioBbox, points, Point2D(bbox.xmin, bbox.ymin));

		//use tiling with distinguish of primary and secondary
	}
}

