#include "KD_tree.h"


KDnode* build(Point2D* points, int num, int depth)
{
	if (num == 0)
	{
		return NULL;
	}
	else if (num == 1)
	{
		KDnode* rlt = new KDnode();
		rlt->leaf = true;
		rlt->num = 1;
		rlt->pts = points;
		//printf("leaf created:\n\n " );
		return rlt;
	}
	else
	{
		KDnode* root = new KDnode();
		root->leaf = false;
		root->num = num;
		root->pts = points;
		char split_axis = (depth % 2) ? Y_AXIS : X_AXIS;
		root->split_axis = split_axis;
		root->cell_bbox.xmax = 0.0; root->cell_bbox.ymax = 0.0;
		root->cell_bbox.xmin = 1.0; root->cell_bbox.ymin = 1.0;

		int num_pos, num_neg;
		if (split_axis == X_AXIS)
		{
			sort(points, points + num, cmp_x);
			//qsort(ps, num, sizeof(Point2D), cmp_x);
			int mid = num >> 1;
			root->split_position = points[mid].x;
			num_pos = mid;
			num_neg = num - mid;
			for (int i = 0; i < num; i++)
			{
				root->cell_bbox.xmax = max(root->cell_bbox.xmax, points[i].x);
				root->cell_bbox.ymax = max(root->cell_bbox.ymax, points[i].y);
				root->cell_bbox.xmin = min(root->cell_bbox.xmin, points[i].x);
				root->cell_bbox.ymin = min(root->cell_bbox.ymin, points[i].y);
			}
			//printf("Node created: %f,%f,%f,%f,%d,%d\n", root->cell_bbox.xmin, root->cell_bbox.ymin, root->cell_bbox.xmax, root->cell_bbox.ymax, num, depth);
			root->left = build(points, num_pos, depth + 1);
			root->right = build(points + num_pos, num_neg, depth + 1);
		}
		else
		{
			sort(points, points + num, cmp_y);
			int mid = num >> 1;
			root->split_position = points[mid].y;
			num_pos = mid;
			num_neg = num - mid;
			for (int i = 0; i < num; i++)
			{
				root->cell_bbox.xmax = max(root->cell_bbox.xmax, points[i].x);
				root->cell_bbox.ymax = max(root->cell_bbox.ymax, points[i].y);
				root->cell_bbox.xmin = min(root->cell_bbox.xmin, points[i].x);
				root->cell_bbox.ymin = min(root->cell_bbox.ymin, points[i].y);
			}

			root->left = build(points, num_pos, depth + 1);
			root->right = build(points + num_pos, num_neg, depth + 1);
		}
		return root;
	}
}


