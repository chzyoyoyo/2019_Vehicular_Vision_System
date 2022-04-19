
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cmath>
#include <float.h>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    
    //VideoCapture cap(1); // 0: default device
    string imageName = "image";
    int imageIdx = 0;
    vector<string> filename;
//    while(1)
//    {
//        Mat frame;
//        cap >> frame;
//        imshow("webcam", frame);
//        int key = waitKey(33);
//        if (key == 27)
//        {
//            char buf[300];
//            sprintf(buf, "output%d.jpg", imageIdx);
//            imwrite(buf, frame);
//            string temp = buf;
//            filename.push_back(temp);
//            imageIdx++;
//        }
//        if (imageIdx == 4)
//            break;
//    }
    
    for(imageIdx = 0; imageIdx < 8; imageIdx++)
    {
        char buf[300];
        sprintf(buf, "/Users/johnsonchen/Desktop/openCV/openCV/output%d.JPG", imageIdx);
        string temp = buf;
        filename.push_back(temp);
    }
    
    vector<vector<Point2f>> allSrcCorner;
    vector<vector<Point3f>> allDstCorner;
    
    Size s;
    int cirsize;
    int i, j, k;
    for (i = 0 ; i < filename.size(); i++)
    {
        Mat image = imread(filename[i], IMREAD_GRAYSCALE);
//        imshow("Image", image);
//        waitKey();
        s = image.size();
        vector<Point2f> srcCorner;
        vector<Point3f> dstCorner;
        for (j = 0; j < 5; j++)
            for (k = 0; k < 7; k++)
                dstCorner.push_back(Point3f(j, k, 0.0f));
        
        for (j = 0; j < filename.size(); j++)
        {
            findChessboardCorners(image, Size(7, 5), srcCorner);
            cornerSubPix(image, srcCorner, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 30, 0.1));
        }
        allSrcCorner.push_back(srcCorner);
        allDstCorner.push_back(dstCorner);
        cout << i << ": " << srcCorner.size() << endl;
        
        for (int a = 0; a < srcCorner.size(); a++)
        {
            circle(image, srcCorner[a], 20, Scalar(255), 2, 8, 0);
        }
        imshow("corner", image);
        waitKey(0);
    }
    
    Mat intrinsic, distCoeffs;
    vector<Mat> rvecs, tvecs;
    calibrateCamera(allDstCorner, allSrcCorner, s, intrinsic, distCoeffs, rvecs,  tvecs);
    
    Mat targetImg = imread("/Users/johnsonchen/Desktop/openCV/openCV/target.JPG", 1);
    Mat finalImg;
    
    Mat R, newIntrinsic;
    Mat outputMapX, outputMapY;
    initUndistortRectifyMap(intrinsic, distCoeffs, Mat(), intrinsic, s, CV_32FC1, outputMapX, outputMapY);
    remap(targetImg, finalImg, outputMapX, outputMapY, INTER_LINEAR);
    
    FileStorage tmp("/Users/johnsonchen/Desktop/openCV/openCV/result.xml", FileStorage::WRITE);
    tmp << "intrinsic" << intrinsic;
    tmp << "distortion" << distCoeffs;
    tmp.release();
    
    imshow("Original Image", targetImg);
    imshow("Final Image", finalImg);
    waitKey();
    return 0;
}
