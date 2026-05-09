#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys =
    "{ help h |                  | Print help message. }"
    "{ input1 | box.png          | Path to input image 1. }"
    "{ input2 | box_in_scene.png | Path to input image 2. }";

int main( int argc, char* argv[] )
{
    Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);
    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    std::string second_argv = argv[2];
    std::string selected_image;

    if(second_argv.find("right08") != std::string::npos)
    {
        selected_image = "right08";
    }
    else
    {
        selected_image = "left10";
    }

    for (size_t i = 0; i < 4; i++)
    {
        Ptr<Feature2D> detector;
        Ptr<Feature2D> descriptor;
        std::string method;
        Ptr<DescriptorMatcher> matcher;

        if (i==0)
        {
            detector = SIFT::create();
            method = "SIFT.png";
            matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        }
        else if (i==1)
        {
            int minHessian = 400;
            detector = SURF::create( minHessian );
            method = "SURF.png";
            matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        }
        else if (i==2)
        {
            detector = ORB::create();
            method = "ORB.png";
            matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
        }
        else
        {
            detector = FastFeatureDetector::create(); 
            descriptor = BriefDescriptorExtractor::create();
            method = "FAST_BRIEF.png";
            matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
        }
        
        std::vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;

        if(i==3)
        {
            detector->detect(img1, keypoints1);
            detector->detect(img2, keypoints2);
            descriptor->compute(img1, keypoints1, descriptors1);
            descriptor->compute(img2, keypoints2, descriptors2);
        }
        else
        {
            detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
            detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );
        }    

        std::vector< std::vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
        const float ratio_thresh = 0.7f;
        std::vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        Mat img_matches;
        drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        Mat img1_descriptors, img2_descriptors;
        drawKeypoints( img1, keypoints1, img1_descriptors, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints( img2, keypoints2, img2_descriptors, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        imwrite("output_task1/left08_" + selected_image + std::string("/img1_descriptors_") + method, img1_descriptors);
        imwrite("output_task1/left08_" + selected_image + std::string("/img2_descriptors_") + method, img2_descriptors);
        imwrite("output_task1/left08_" + selected_image + std::string("/correspondences_") + method, img_matches);
    }

    return 0;
}
#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif