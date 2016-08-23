#include "facedetect-dll.h"
#include <vector>
#include "opencv2/opencv.hpp"

void facedetect(const cv::Mat_<uchar> &image, std::vector<cv::Rect> &bboxs)
{
	int * pResults = NULL;
	//pResults = facedetect_frontal_tmp((unsigned char*)(tmp.ptr(0)), tmp.cols, tmp.rows, tmp.step, 1.2f, 5, 24);
	//pResults = facedetect_frontal((unsigned char*)(tmp.ptr(0)), tmp.cols, tmp.rows, tmp.step, 1.2f, 3, 24);
	//pResults = facedetect_multiview((unsigned char*)(tmp.ptr(0)), tmp.cols, tmp.rows, tmp.step, 1.2f, 5, 24);
	pResults = facedetect_multiview_reinforce((unsigned char*)(image.ptr(0)), image.cols, image.rows, image.step, 1.2f, 5, 24);

	int facenum = (pResults ? *pResults : 0);
	for (int i = 0; i < facenum; i++)
	{
		short * p = ((short*)(pResults + 1)) + 6 * i;
		cv::Rect t;
		t.x = p[0];
		t.y = p[1];
		t.width = p[2];
		t.height = p[3];
		//int neighbors = p[4];
		//int angle = p[5];
		bboxs.push_back(t);
	}
}

