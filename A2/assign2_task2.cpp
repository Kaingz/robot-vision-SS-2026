#include <iostream>
#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include <numeric>
#include <random>

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys =
    "{ help h |                  | Print help message. }"
    "{ input1 | box.png          | Path to input image 1. }"
    "{ input2 | box_in_scene.png | Path to input image 2. }";

void drawLinesAndSaveImg(std::vector<Point2f>& pts1, std::vector<Point2f>& pts2, Mat F, Mat img1, Mat img2, std::string filename, const Mat& mask = cv::Mat());

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
        std::vector<Point2f> pts1;
        std::vector<Point2f> pts2;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);

                // Task2
                pts1.push_back(keypoints1[good_matches.back().queryIdx].pt);
                pts2.push_back(keypoints2[good_matches.back().trainIdx].pt);
            }
        }

        Mat mask_8P_Ransac;
        Mat mask_5P_Ransac;

        Mat F_8P = findFundamentalMat(pts1, pts2, FM_8POINT);
        Mat F_8P_Ransac = findFundamentalMat(pts1, pts2, FM_RANSAC, 3, 0.99, mask_8P_Ransac);

        Mat K_left = (Mat_<double>(3, 3) <<
            9.842439e+02,  0.000000e+00, 6.900000e+02,
            0.000000e+00, 9.808141e+02, 2.331966e+02,
            0.000000e+00, 0.000000e+00, 1.000000e+00
        );

        Mat K_right;

        if(selected_image == "right08")
        {
            K_right = (Mat_<double>(3, 3) <<
                9.895267e+02, 0.000000e+00, 7.020000e+02,
                0.000000e+00, 9.878386e+02, 2.455590e+02,
                0.000000e+00, 0.000000e+00, 1.000000e+00
            );
        }
        else
        {
            K_right = K_left;
        }

        std::vector<Point2f> pts1_norm;
        std::vector<Point2f> pts2_norm;

        undistortPoints(pts1, pts1_norm, K_left, cv::noArray());
        undistortPoints(pts2, pts2_norm, K_right, cv::noArray());

        Mat E_5P_Ransac = findEssentialMat(pts1_norm, pts2_norm, Mat::eye(3, 3, CV_64F), RANSAC, 0.99, 1e-3, mask_5P_Ransac);

        Mat F_5P_Ransac = K_right.inv().t() * E_5P_Ransac * K_left.inv();

        std::string output_path = "output_task2/left08_" + selected_image;

        drawLinesAndSaveImg(pts1, pts2, F_8P, img1, img2, std::string(output_path + "_8P_All_" + method));
        drawLinesAndSaveImg(pts1, pts2, F_8P_Ransac, img1, img2, std::string(output_path + "_8P_Ransac_" + method));
        drawLinesAndSaveImg(pts1, pts2, F_5P_Ransac, img1, img2, std::string(output_path + "_5P_Ransac_" + method));
    }

    return 0;
}

void drawLinesAndSaveImg(std::vector<Point2f>& pts1, std::vector<Point2f>& pts2, Mat F, Mat img1, Mat img2, std::string filename, const Mat& mask)
{
    std::vector<Vec3f> lines_left;
    computeCorrespondEpilines(pts2, 2, F, lines_left);
    std::vector<Vec3f> lines_right;
    computeCorrespondEpilines(pts1, 1, F, lines_right);

    cvtColor(img1, img1, COLOR_GRAY2BGR);
    cvtColor(img2, img2, COLOR_GRAY2BGR);

    std::vector<int> random_idx(pts1.size());
    std::iota(random_idx.begin(), random_idx.end(), 0);

    std::mt19937 g(std::random_device{}());
    std::shuffle(random_idx.begin(), random_idx.end(), g);
    std::uniform_int_distribution dist(0,255);

    uint successful_draws = 0;
    uint i = 0;

    while(successful_draws < 20 && random_idx.size() > 0)
    {
        int idx = random_idx.back();
        random_idx.pop_back();

        if(!mask.empty() && mask.at<uchar>(idx) == 0)
        {
            i++;
            continue;
        }

        successful_draws++;
        
        cv::Scalar random_color(dist(g), dist(g), dist(g));

        Vec3f& l = lines_left[idx];
        uint c = img1.cols;

        Point p1(0, -l[2]/l[1]);
        Point p2(c, -(l[2] + l[0]*c)/l[1]);
        line(img1, p1, p2, random_color, 1);
        circle(img1, pts1[idx],5,random_color, -1);

        l = lines_right[idx];
        c = img2.cols;

        p1 = Point(0, -l[2]/l[1]);
        p2 = Point(c, -(l[2] + l[0]*c)/l[1]);
        line(img2, p1, p2, random_color, 1);
        circle(img2, pts2[idx],5,random_color, -1);
    }

    Mat img_pair;

    hconcat(img1, img2, img_pair);

    imwrite(filename, img1);
}

#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif