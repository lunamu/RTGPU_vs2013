
#include "UtilKernels.cuh"

void viewGPUPoint2D(Point2D* points, int num, string filename)
{
	const string dir = "C:/Users/lunam/Dropbox/MATLAB/";
	Point2D* host_array = new Point2D[num];
	gpuErrchk(cudaMemcpy(host_array, points, sizeof(Point2D) * num, cudaMemcpyDeviceToHost));
	ofstream file(dir + filename);
	for (int i = 0; i < num; i++)
	{
		file << host_array[i].x << " " << host_array[i].y << endl;
	}
	file.close();
	delete[] host_array;
}


void viewGPUPoint2D(PointStruct* points, int num, string filename)
{
	const string dir = "C:/Users/lunam/Dropbox/MATLAB/";
	PointStruct* host_array = new PointStruct[num];
	gpuErrchk(cudaMemcpy(host_array, points, sizeof(PointStruct) * num, cudaMemcpyDeviceToHost));
	ofstream file(dir + filename);
	for (int i = 0; i < num; i++)
	{
		file << host_array[i].p.x << " " << host_array[i].p.y << endl;
	}
	file.close();
	delete[] host_array;
}