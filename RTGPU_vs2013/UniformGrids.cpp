#include "UniformGrids.h"


UniformGrids::UniformGrids(int dimension)
{
	dimension_edge = dimension;
	width = dimension;
	height = dimension;
	grids = new Grid[dimension_edge * dimension_edge];

	gridBbox = gUnitGrid;
	w_f = gridBbox.xmax - gridBbox.xmin;
	h_f = gridBbox.ymax - gridBbox.ymin;
	itval = 1.0 / dimension_edge;
}

//create with width and height
UniformGrids::UniformGrids(int w, int h, BBox _bbox)
{
	width = w;
	height = h;
	grids = new Grid[width * height];
	gridBbox = _bbox;
	w_f = _bbox.xmax - _bbox.xmin;
	h_f = _bbox.ymax - _bbox.ymin;
	itval_w = (_bbox.xmax - _bbox.xmin) / w;
	itval_h = (_bbox.ymax - _bbox.ymin) / h;
}

void UniformGrids::insert(Point2D point, int priority)
{
	if (!within(gridBbox, point))return;
	unsigned indx = SpatialToIdx(point);
	if (grids[indx].num >= MAX_PER_GRID)
	{
		printf("full grid error, indx %d\n", indx);
		return;
	}
	grids[indx].points[grids[indx].num] = point;
	grids[indx].valid[grids[indx].num] = VALID;
	grids[indx].priority[grids[indx].num] = priority;
	grids[indx].num++;

}

//return false if there are some points within range.
//otherwise return true;
bool UniformGrids::dartSearch(Point2D point, double range)
{
	unsigned int x_start = max(0, (point.x - range - gridBbox.xmin)) * width;
	unsigned int x_end = min(gridBbox.xmax - EPS - gridBbox.xmin, (point.x + range - gridBbox.xmin)) * width;
	unsigned int y_start = max(0, (point.y - range - gridBbox.ymin)) * height;
	unsigned int y_end = min(gridBbox.ymax - EPS - gridBbox.ymin, (point.y + range - gridBbox.ymin)) * height;
	for (int i = x_start; i <= x_end; i++)
	{
		for (int j = y_start; j <= y_end; j++)
		{
			unsigned int idx = width * j + i;
			if (grids[idx].num <= 0)
			{
				continue;
			}
			else
			{
				for (int i = 0; i < grids[idx].num; i++)
				{
					if (!grids[idx].valid[i])continue;
					double distSquare = DistanceSquared(point, grids[idx].points[i]);
					if (distSquare == 0.0)continue;
					//if (distSquare < 1e-20)continue;
					else if (distSquare < range * range)
					{
						return false;
					}
				}
			}
		}
	}
	return true;
}

bool UniformGrids::dartSearch(Point2D point, double range, bool& same)
{
	unsigned int x_start = max(0, (point.x - range - gridBbox.xmin)) * width;
	unsigned int x_end = min(gridBbox.xmax - EPS - gridBbox.xmin, (point.x + range - gridBbox.xmin)) * width;
	unsigned int y_start = max(0, (point.y - range - gridBbox.ymin)) * height;
	unsigned int y_end = min(gridBbox.ymax - EPS - gridBbox.ymin, (point.y + range - gridBbox.ymin)) * height;
	same = false;
	for (int i = x_start; i <= x_end; i++)
	{
		for (int j = y_start; j <= y_end; j++)
		{
			unsigned int idx = width * j + i;
			if (grids[idx].num <= 0)
			{
				continue;
			}
			else
			{
				for (int i = 0; i < grids[idx].num; i++)
				{
					if (!grids[idx].valid[i])continue;
					double distSquare = DistanceSquared(point, grids[idx].points[i]);
					if (distSquare == 0.0) {
						same = true; continue;
					}
					//if (distSquare < 1e-20)continue;
					else if (distSquare < range * range)
					{
						
						return false;
					}
				}
			}
		}
	}
	return true;
}

//push all points withing the ring (from inner range to outer range) to the buffer buf
void UniformGrids::ringSearch_buffer(Point2D point, double range_inner, double range_outer, vector<Point2D>& buf)
{
	unsigned int x_start = max(0, (point.x - range_outer - gridBbox.xmin)) * width;
	unsigned int x_end = min(gridBbox.xmax - EPS - gridBbox.xmin, (point.x + range_outer - gridBbox.xmin)) * width;
	unsigned int y_start = max(0, (point.y - range_outer - gridBbox.ymin)) * height;
	unsigned int y_end = min(gridBbox.ymax - EPS - gridBbox.ymin, (point.y + range_outer - gridBbox.ymin)) * height;
	for (int x = x_start; x <= x_end; x++)
	{
		for (int y = y_start; y <= y_end; y++)
		{
			unsigned int idx = dimension_edge * y + x;
			if (grids[idx].num <= 0)
			{
				continue;
			}
			else
			{
				for (int i = 0; i < grids[idx].num; i++)
				{
					if (!grids[idx].valid[i])continue;

					if (DistanceSquared(point, grids[idx].points[i]) == 0.0)continue;

					else if ((DistanceSquared(point, grids[idx].points[i]) < range_outer * range_outer) && (DistanceSquared(point, grids[idx].points[i]) >  range_inner * range_inner))
					{
						buf.push_back(grids[idx].points[i]);
					}
				}
			}
		}
	}
}

//push all conflict points to conflictbuffer, record their priority, grididx and in_grid idx;
void UniformGrids::dartSearch_buffer_pri(Point2D point, double range, vector<Point2D>& conflictBuffer, vector<int>& conflictPri, vector<int>& conflictGrididx, vector<int>& conflictIngrid_idx)
{
	unsigned int x_start = max(0, (point.x - range - gridBbox.xmin)) * width;
	unsigned int x_end = min(gridBbox.xmax - EPS - gridBbox.xmin, (point.x + range - gridBbox.xmin)) * width;
	unsigned int y_start = max(0, (point.y - range - gridBbox.ymin)) * height;
	unsigned int y_end = min(gridBbox.ymax - EPS - gridBbox.ymin, (point.y + range - gridBbox.ymin)) * height;
	for (int i = x_start; i <= x_end; i++)
	{
		for (int j = y_start; j <= y_end; j++)
		{
			unsigned int idx = dimension_edge * j + i;
			if (grids[idx].num <= 0)
			{
				continue;
			}
			else
			{
				for (int i = 0; i < grids[idx].num; i++)
				{
					if (!grids[idx].valid[i])continue;

					if (DistanceSquared(point, grids[idx].points[i]) == 0.0)continue;

					else if (DistanceSquared(point, grids[idx].points[i]) < range * range)
					{
						conflictBuffer.push_back(grids[idx].points[i]);
						conflictPri.push_back(grids[idx].priority[i]);
						conflictGrididx.push_back(idx);
						conflictIngrid_idx.push_back(i);
					}
				}
			}
		}
	}
}

void UniformGrids:: dartSearch_buffer(Point2D point, double range, vector<Point2D>& buf)
{
	unsigned int x_start = max(0, (point.x - range - gridBbox.xmin)) * width;
	unsigned int x_end = min(gridBbox.xmax - EPS - gridBbox.xmin, (point.x + range - gridBbox.xmin)) * width;
	unsigned int y_start = max(0, (point.y - range - gridBbox.ymin)) * height;
	unsigned int y_end = min(gridBbox.ymax - EPS - gridBbox.ymin, (point.y + range - gridBbox.ymin)) * height;
	for (int i = x_start; i <= x_end; i++)
	{
		for (int j = y_start; j <= y_end; j++)
		{
			unsigned int idx = width * j + i;
			if (grids[idx].num <= 0)
			{
				continue;
			}
			else
			{
				for (int i = 0; i < grids[idx].num; i++)
				{
					if (!grids[idx].valid[i])continue;
					double distSquare = DistanceSquared(point, grids[idx].points[i]);
					if (distSquare == 0.0)continue;
					//if (distSquare < 1e-20)continue;
					else if (distSquare < range * range)
					{
						buf.push_back(grids[idx].points[i]);
					}
				}
			}
		}
	}
}


void UniformGrids::insert_in_gap(Point2D pivotPoint, Point2D conflictPoint, double radius)
{
	vector<Point2D> pointTestBuffer;
	ringSearch_buffer(conflictPoint, 2 * radius, 4 * radius, pointTestBuffer);
	//cout << "PointTestBuffer:" << endl;
	//for (int i = 0; i < pointTestBuffer.size(); i++)
	//{
	//	cout << pointTestBuffer[i].x << " " << pointTestBuffer[i].y << endl;
	//}
	//Here let's try 
	//First, fill all from pivot point
	for (auto first_point = pointTestBuffer.begin(); first_point != pointTestBuffer.end(); first_point++)
	{
		for (auto second_point = first_point + 1; second_point != pointTestBuffer.end(); second_point++)
		{
			if (first_point == second_point) continue;
			auto fp = *first_point;
			auto sp = *second_point;
			Point2D center;
			double cir_r2;
			Circumcenter(pivotPoint, fp, sp, center, cir_r2);
			//if (cir_r2 > 16 * radius * radius)continue;
			if (center.x > gridBbox.xmax || center.x < gridBbox.xmin || center.y > gridBbox.ymax || center.y < gridBbox.ymin)continue;
			else if (dartSearch(center, 2 * radius))
			{
				insert(center);
			}

		}
	}


	//for (auto first_point = pointTestBuffer.begin(); first_point != pointTestBuffer.end(); first_point++)
	//{
	//	for (auto second_point = first_point + 1; second_point != pointTestBuffer.end(); second_point++)
	//	{
	//		for (auto third_point = second_point + 1; third_point != pointTestBuffer.end(); third_point++)
	//		{
	//			if (first_point == second_point || first_point == third_point || second_point == third_point) continue;
	//			auto fp = *first_point;
	//			auto sp = *second_point;
	//			auto tp = *third_point;
	//			Point2D center;
	//			double cir_r2;
	//			Circumcenter(fp, sp, tp, center, cir_r2);
	//			if (cir_r2 > 16 * radius * radius)continue;
	//			if (center.x > 1 || center.x < 0 || center.y > 1 || center.y < 0)continue;
	//			else if (dartSearch(center, 2 * radius))
	//			{
	//				insert(center);
	//				/*for (auto fp2 = pointTestBuffer.begin(); fp2 != pointTestBuffer.end(); fp2++)
	//				{
	//					for (auto sp2 = fp2 + 1; sp2 != pointTestBuffer.end(); sp2++)
	//					{
	//						if (fp2 == sp2) continue;
	//						auto fp2_d = *first_point;
	//						auto sp2_d = *second_point;
	//						Point2D center2;
	//						double cir_r2;
	//						Circumcenter(center, fp2_d, sp2_d, center2, cir_r2);
	//						if (cir_r2 > 16 * range * range)continue;
	//						if (center2.x > 1 || center2.x < 0 || center2.y > 1 || center2.y < 0)continue;
	//						else if (dartSearch(center2, range))
	//						{
	//							insert(center2);
	//						}

	//					}
	//				}*/
	//			}
	//		}
	//	}
	//}
	//pointTestBuffer.clear();
}


void UniformGrids::process_pivot_point(Point2D& pivotPoint, int pri, double radius, int grid_idx, int ingrid_idx)
{
	vector<Point2D> conflictBuffer;
	vector<int> conflictPri;
	vector<int> conflictGrididx;
	vector<int> conflictIngrid_idx;
	dartSearch_buffer_pri(pivotPoint, 2 * radius, conflictBuffer, conflictPri, conflictGrididx, conflictIngrid_idx);
	for (int i = 0; i < conflictBuffer.size(); i++)
	{
		if (conflictPri[i] > pri)
		{
			process_pivot_point(conflictBuffer[i], conflictPri[i], radius, conflictGrididx[i], conflictIngrid_idx[i]);
			return;
		}
	}
	//cout << "Pivot point: " << pivotPoint.x << " " << pivotPoint.y << endl;
	//cout << "Conflict point: " << endl;
	/*for (int i = 0; i < conflictBuffer.size(); i++)
	{
	cout << conflictBuffer[i].x << " " << conflictBuffer[i].y << endl;
	}*/
	for (int i = 0; i < conflictBuffer.size(); i++)
	{
		int gidx = conflictGrididx[i];
		int iidx = conflictIngrid_idx[i];
		grids[gidx].valid[iidx] = INVALID;	//eliminate;
		insert_in_gap(pivotPoint, conflictBuffer[i], radius);
	}
}



void UniformGrids::eliminate_for_maximal(double radius)
{
	for (int grid_idx = 0; grid_idx < width * height; grid_idx++)
	{
		if (grids[grid_idx].num > 0)
		{
			for (int in_idx = 0; in_idx < grids[grid_idx].num; in_idx++)//iteration of all points
			{
				Point2D& cur_point = grids[grid_idx].points[in_idx];
				int cur_pri = grids[grid_idx].priority[in_idx];

				process_pivot_point(cur_point, cur_pri, radius, grid_idx, in_idx);
			}
		}
	}
}

template<typename T>
bool not_in(vector<T> vec, T t)
{
	for (int i = 0; i < vec.size(); i++)
	{
		if (vec[i] == t)return false;
	}
	return true;
}


void UniformGrids::insert_in_gap_batch(Point2D pivotPoint, vector<Point2D> conflictPoint, double radius)
{
	vector<Point2D> pointTestBuffer;
	for (int i = 0; i < conflictPoint.size(); i++)
	{
		vector<Point2D> tmp_test_buffer;
		//ringSearch_buffer(conflictPoint[i], 2 * radius, 4 * radius, tmp_test_buffer);
		dartSearch_buffer(conflictPoint[i],  4 * radius, tmp_test_buffer);
		for (Point2D cfp : tmp_test_buffer)
		{
			if (not_in<Point2D>(pointTestBuffer, cfp))pointTestBuffer.push_back(cfp);
		}
	}

	for (auto first_point = pointTestBuffer.begin(); first_point != pointTestBuffer.end(); first_point++)
	{
		for (auto second_point = first_point + 1; second_point != pointTestBuffer.end(); second_point++)
		{
			for (auto third_point = second_point + 1; third_point != pointTestBuffer.end(); third_point++)
			{
				if (first_point == second_point || first_point == third_point || second_point == third_point) continue;
				auto fp = *first_point;
				auto sp = *second_point;
				auto tp = *third_point;
				Point2D center;
				double cir_r2;
				Circumcenter(fp, sp, tp, center, cir_r2);
				if (cir_r2 > 16 * radius * radius)continue;
				if (center.x > 1 || center.x < 0 || center.y > 1 || center.y < 0)continue;
				else if (dartSearch(center, 2 * radius))
				{
					insert(center);
					/*for (auto fp2 = pointTestBuffer.begin(); fp2 != pointTestBuffer.end(); fp2++)
					{
						for (auto sp2 = fp2 + 1; sp2 != pointTestBuffer.end(); sp2++)
						{
							if (fp2 == sp2) continue;
							auto fp2_d = *first_point;
							auto sp2_d = *second_point;
							Point2D center2;
							double cir_r2;
							Circumcenter(center, fp2_d, sp2_d, center2, cir_r2);
							if (cir_r2 > 16 * range * range)continue;
							if (center2.x > 1 || center2.x < 0 || center2.y > 1 || center2.y < 0)continue;
							else if (dartSearch(center2, range))
							{
								insert(center2);
							}

						}
					}*/
				}
			}
		}
	}


	pointTestBuffer.clear();
}

void UniformGrids::process_pivot_point_batch(Point2D& pivotPoint, int pri, double radius, int grid_idx, int ingrid_idx)
{
	vector<Point2D> conflictBuffer;
	vector<int> conflictPri;
	vector<int> conflictGrididx;
	vector<int> conflictIngrid_idx;
	dartSearch_buffer_pri(pivotPoint, 2 * radius, conflictBuffer, conflictPri, conflictGrididx, conflictIngrid_idx);
	for (int i = 0; i < conflictBuffer.size(); i++)
	{
		if (conflictPri[i] > pri)
		{
			process_pivot_point_batch(conflictBuffer[i], conflictPri[i], radius, conflictGrididx[i], conflictIngrid_idx[i]);
			return;
		}
	}
	//cout << "Pivot point: " << pivotPoint.x << " " << pivotPoint.y << endl;
	//cout << "Conflict point: " << endl;
	/*for (int i = 0; i < conflictBuffer.size(); i++)
	{
	cout << conflictBuffer[i].x << " " << conflictBuffer[i].y << endl;
	}*/
	for (int i = 0; i < conflictBuffer.size(); i++)
	{
		int gidx = conflictGrididx[i];
		int iidx = conflictIngrid_idx[i];
		grids[gidx].valid[iidx] = INVALID;	//eliminate;
	}
	insert_in_gap_batch(pivotPoint, conflictBuffer, radius);
}




void UniformGrids::eliminate_for_maximal_batch(double radius)
{
	for (int grid_idx = 0; grid_idx < width * height; grid_idx++)
	{
		if (grids[grid_idx].num > 0)
		{
			for (int in_idx = 0; in_idx < grids[grid_idx].num; in_idx++)//iteration of all points
			{
				Point2D& cur_point = grids[grid_idx].points[in_idx];
				int cur_pri = grids[grid_idx].priority[in_idx];

				process_pivot_point_batch(cur_point, cur_pri, radius, grid_idx, in_idx);
			}
		}
	}
}



void normalize(Point2D& vec)
{
	if (vec.x == 0 && vec.y == 0)
	{
		vec.x = 1; vec.y = 0; return;
	}
	else
	{
		float len = sqrtf(vec.x * vec.x + vec.y * vec.y);
		vec.x /= len;
		vec.y /= len;
	}
}

void circle_intersection(Point2D c1, Point2D c2, Point2D& p1, Point2D& p2, double radius)
{
	float d_squared = DistanceSquared(c1, c2);
	float half_arc = sqrt(radius * radius - 0.25 * d_squared);
	Point2D vec_c1_c2(c2.x - c1.x, c2.y - c1.y);
	Point2D vec_vertical_c1_c2(-(c2.y - c1.y), c2.x - c1.x);
	normalize(vec_vertical_c1_c2);

	p1.x = c1.x + 0.5 * vec_c1_c2.x + half_arc * vec_vertical_c1_c2.x;
	p1.y = c1.y + 0.5 * vec_c1_c2.y + half_arc * vec_vertical_c1_c2.y;

	p2.x = c1.x + 0.5 * vec_c1_c2.x - half_arc * vec_vertical_c1_c2.x;
	p2.y = c1.y + 0.5 * vec_c1_c2.y - half_arc * vec_vertical_c1_c2.y;
}


void UniformGrids::insert_in_gap_corner(Point2D pivotPoint, Point2D conflictPoint, double radius)
{
	vector<Point2D> pointTestBuffer;
	ringSearch_buffer(conflictPoint, 2 * radius, 4 * radius, pointTestBuffer);
	
	pointTestBuffer.push_back(pivotPoint);

	vector<Point2D> cornerPoints;
	vector<bool> valid;
	for (auto first_point = pointTestBuffer.begin(); first_point != pointTestBuffer.end(); first_point++)
	{
		for (auto second_point = first_point + 1; second_point != pointTestBuffer.end(); second_point++)
		{
			if (first_point == second_point) continue;
			auto fp = *first_point;
			auto sp = *second_point;
			Point2D center;
			double cir_r2;
			Point2D p1, p2;
			circle_intersection(fp, sp, p1, p2, 2*radius);

			//Unfinished, make it in different functions.
			double eps = 0.001 * radius * radius;
			bool corner = true;
			for (Point2D pt : pointTestBuffer)
			{
				if (DistanceSquared(pt, p1) < 4 * radius * radius + eps) {
					corner = false; break;
				}
			}
			if(corner) { cornerPoints.push_back(p1); valid.push_back(true); }

			corner = true;
			for (Point2D pt : pointTestBuffer)
			{
				if (DistanceSquared(pt, p2) < 4 * radius * radius + eps) {
					corner = false; break;
				}
			}
			if(corner) { cornerPoints.push_back(p2); valid.push_back(true); }


			//if (cir_r2 > 16 * radius * radius)continue;
			if (center.x > gridBbox.xmax || center.x < gridBbox.xmin || center.y > gridBbox.ymax || center.y < gridBbox.ymin)continue;
			else if (dartSearch(center, 2 * radius))
			{
				insert(center);
			}

		}
	}


	for (auto first_point = pointTestBuffer.begin(); first_point != pointTestBuffer.end(); first_point++)
	{
		for (auto second_point = first_point + 1; second_point != pointTestBuffer.end(); second_point++)
		{
			for (auto third_point = second_point + 1; third_point != pointTestBuffer.end(); third_point++)
			{
				if (first_point == second_point || first_point == third_point || second_point == third_point) continue;
				auto fp = *first_point;
				auto sp = *second_point;
				auto tp = *third_point;
				Point2D center;
				double cir_r2;
				Circumcenter(fp, sp, tp, center, cir_r2);
				if (cir_r2 > 16 * radius * radius)continue;
				if (center.x > 1 || center.x < 0 || center.y > 1 || center.y < 0)continue;
				else if (dartSearch(center, 2 * radius))
				{
					insert(center);
					/*for (auto fp2 = pointTestBuffer.begin(); fp2 != pointTestBuffer.end(); fp2++)
					{
					for (auto sp2 = fp2 + 1; sp2 != pointTestBuffer.end(); sp2++)
					{
					if (fp2 == sp2) continue;
					auto fp2_d = *first_point;
					auto sp2_d = *second_point;
					Point2D center2;
					double cir_r2;
					Circumcenter(center, fp2_d, sp2_d, center2, cir_r2);
					if (cir_r2 > 16 * range * range)continue;
					if (center2.x > 1 || center2.x < 0 || center2.y > 1 || center2.y < 0)continue;
					else if (dartSearch(center2, range))
					{
					insert(center2);
					}

					}
					}*/
				}
			}
		}
	}
	pointTestBuffer.clear();
}



void UniformGrids::process_pivot_point_corner(Point2D& pivotPoint, int pri, double radius, int grid_idx, int ingrid_idx)
{
	vector<Point2D> conflictBuffer;
	vector<int> conflictPri;
	vector<int> conflictGrididx;
	vector<int> conflictIngrid_idx;
	
	vector<Point2D> gap_corner_points;
	//vector<Point2D> involved_points;

	dartSearch_buffer_pri(pivotPoint, 2 * radius, conflictBuffer, conflictPri, conflictGrididx, conflictIngrid_idx);
	for (int i = 0; i < conflictBuffer.size(); i++)
	{
		if (conflictPri[i] > pri)
		{
			process_pivot_point(conflictBuffer[i], conflictPri[i], radius, conflictGrididx[i], conflictIngrid_idx[i]);
			return;
		}
	}
	//cout << "Pivot point: " << pivotPoint.x << " " << pivotPoint.y << endl;
	//cout << "Conflict point: " << endl;
	/*for (int i = 0; i < conflictBuffer.size(); i++)
	{
	cout << conflictBuffer[i].x << " " << conflictBuffer[i].y << endl;
	}*/
	for (int i = 0; i < conflictBuffer.size(); i++)
	{
		int gidx = conflictGrididx[i];
		int iidx = conflictIngrid_idx[i];
		grids[gidx].valid[iidx] = INVALID;	//eliminate;
		insert_in_gap_corner(pivotPoint, conflictBuffer[i], radius);
	}


}


void UniformGrids::eliminate_for_maximal_corner(double radius)
{
	for (int grid_idx = 0; grid_idx < width * height; grid_idx++)
	{
		if (grids[grid_idx].num > 0)
		{
			for (int in_idx = 0; in_idx < grids[grid_idx].num; in_idx++)//iteration of all points
			{
				Point2D& cur_point = grids[grid_idx].points[in_idx];
				int cur_pri = grids[grid_idx].priority[in_idx];

				process_pivot_point_corner(cur_point, cur_pri, radius, grid_idx, in_idx);
			}
		}
	}
}


void UniformGrids::print()
{
	for (int i = 0; i < dimension_edge * dimension_edge; i++)
	{
		if (grids[i].num > 1)
			printf("%d, %f, %f\n", grids[i].num, grids[i].points[0].x, grids[i].points[0].y);
	}
}

void UniformGrids::test_maximal(double radius)
{
	//for each valid point in the grid, search range 4*r
	//test if circum center within 2*r from any points.
	ofstream gap_file("C:/Users/mu/Dropbox/MATLAB/gaps.txt");
	for (int grid_idx = 0; grid_idx < width * height; grid_idx++)
	{
		for (int in_idx = 0; in_idx < grids[grid_idx].num; in_idx++)//iteration of all points
		{
			if (grids[grid_idx].valid[in_idx])
			{
				Point2D& cur_point = grids[grid_idx].points[in_idx];
				vector<Point2D> conflictBuffer;
				vector<int> conflictPri;
				vector<int> conflictGrididx;
				vector<int> conflictIngrid_idx;
				dartSearch_buffer_pri(cur_point, 4 * radius, conflictBuffer, conflictPri, conflictGrididx, conflictIngrid_idx);

				for (auto first_point = conflictBuffer.begin(); first_point != conflictBuffer.end(); first_point++)
				{
					for (auto second_point = first_point + 1; second_point != conflictBuffer.end(); second_point++)
					{
						if (first_point == second_point) continue;
						auto fp = *first_point;
						auto sp = *second_point;
						Point2D center;
						double cir_r2;
						Circumcenter(cur_point, fp, sp, center, cir_r2);
						//if (cir_r2 > 16 * range * range)continue;
						bool same;
						if (center.x > gridBbox.xmax || center.x < gridBbox.xmin || center.y > gridBbox.ymax || center.y < gridBbox.ymin)continue;
						else if (dartSearch(center, 2 * radius, same))
						{
							if(!same)gap_file << center.x << " " << center.y << endl;
						}

					}
				}
			}

		}



	}
	printf("done, gaps above\n");


}

UniformGrids::~UniformGrids()
{
	delete[] grids;
}