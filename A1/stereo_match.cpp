/*
 *  stereo_match.cpp
 *  calibration
 *
 *  Created by Victor  Eruhimov on 1/18/10.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <sstream>

using namespace cv;

static void print_help(char** argv)
{
    printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    printf("\nUsage: %s <left_image> <right_image> [--algorithm=bm|sgbm|hh|hh4|sgbm3way] [--blocksize=<block_size>]\n"
           "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
           "[--no-display] [--color] [-o=<disparity_image>] [-p=<point_cloud_file>]\n", argv[0]);
}

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

static void savePLY(const char* filename, const Mat& mat, const Mat& img)
{
    const double max_z = 1.0e4;
    
    //Count  valid points 
    int valid_points = 0;
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            // Skip points that are infinitely far away (invalid disparity)
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            valid_points++;
        }
    }

    FILE* fp = fopen(filename, "wt");
    if(!fp) 
    {
        printf("Error: Could not open %s for writing.\n", filename);
        return;
    }

    //PLY Header
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", valid_points);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");

    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            
            Vec3b color = img.at<Vec3b>(y, x);
            
            uchar b = color[0];
            uchar g = color[1];
            uchar r = color[2];
            fprintf(fp, "%f %f %f %d %d %d\n", point[0], point[1], point[2], r, g, b);
        }
    }
    
    fclose(fp);
    printf("Successfully saved %d colored points to %s\n", valid_points, filename);
}

int main(int argc, char** argv)
{
    std::string img1_filename = "";
    std::string img2_filename = "";
    std::string intrinsic_filename = "";
    std::string extrinsic_filename = "";
    std::string disparity_filename = "";
    std::string point_cloud_filename = "";

    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4, STEREO_HH4=5 };
    int alg = STEREO_SGBM;
    int SADWindowSize, numberOfDisparities;
    bool no_display;
    bool color_display;
    float scale;

    Ptr<StereoBM> bm = StereoBM::create(16,9);
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
    cv::CommandLineParser parser(argc, argv,
        "{@arg1||}{@arg2||}{help h||}{algorithm||}{max-disparity|0|}{blocksize|0|}{no-display||}{color||}{scale|1|}{i||}{e||}{o||}{p||}");
    if(parser.has("help"))
    {
        print_help(argv);
        return 0;
    }
    img1_filename = samples::findFile(parser.get<std::string>(0));
    img2_filename = samples::findFile(parser.get<std::string>(1));
    if (parser.has("algorithm"))
    {
        std::string _alg = parser.get<std::string>("algorithm");
        alg = _alg == "bm" ? STEREO_BM :
            _alg == "sgbm" ? STEREO_SGBM :
            _alg == "hh" ? STEREO_HH :
            _alg == "var" ? STEREO_VAR :
            _alg == "hh4" ? STEREO_HH4 :
            _alg == "sgbm3way" ? STEREO_3WAY : -1;
    }
    numberOfDisparities = parser.get<int>("max-disparity");
    SADWindowSize = parser.get<int>("blocksize");
    scale = parser.get<float>("scale");
    no_display = parser.has("no-display");
    color_display = parser.has("color");
    if( parser.has("i") )
        intrinsic_filename = parser.get<std::string>("i");
    if( parser.has("e") )
        extrinsic_filename = parser.get<std::string>("e");
    if( parser.has("o") )
        disparity_filename = parser.get<std::string>("o");
    if( parser.has("p") )
        point_cloud_filename = parser.get<std::string>("p");
    if (!parser.check())
    {
        parser.printErrors();
        return 1;
    }
    if( alg < 0 )
    {
        printf("Command-line parameter error: Unknown stereo algorithm\n\n");
        print_help(argv);
        return -1;
    }
    if ( numberOfDisparities < 1 || numberOfDisparities % 16 != 0 )
    {
        printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
        print_help(argv);
        return -1;
    }
    if (scale < 0)
    {
        printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
        return -1;
    }
    // if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
    // {
    //     printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
    //     return -1;
    // }
    if( img1_filename.empty() || img2_filename.empty() )
    {
        printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }
    if( (!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()) )
    {
        printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
        return -1;
    }

    if( extrinsic_filename.empty() && !point_cloud_filename.empty() )
    {
        printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
        return -1;
    }

    Mat img1 = imread(img1_filename, cv::IMREAD_COLOR);
    Mat img2 = imread(img2_filename, cv::IMREAD_COLOR);

    if (img1.empty())
    {
        printf("Command-line parameter error: could not load the first input image file\n");
        return -1;
    }
    if (img2.empty())
    {
        printf("Command-line parameter error: could not load the second input image file\n");
        return -1;
    }

    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }

    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;

    if( !intrinsic_filename.empty() )
    {
        // reading intrinsic parameters
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename.c_str());
            return -1;
        }

        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        fs.open(extrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename.c_str());
            return -1;
        }

        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;

        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;

        imwrite("output_task3/left/rectified_left.png", img1);
        imwrite("output_task3/right/rectified_right.png", img2);
    }


std::string py_command = "python3 unimatch/main_stereo.py "
                             "--checkpoint_dir /tmp "
                             "--inference_dir_left output_task3/left "
                             "--inference_dir_right output_task3/right "
                             "--output_path output_task3/unimatch_output "
                             "--inference_size 864 1536 "
                             "--num_scales 2 "
                             "--reg_refine "
                             "--num_reg_refine 3 "
                             "--upsample_factor 4 "
                             "--attn_splits_list 2 8 "    
                             "--corr_radius_list -1 4 "   
                             "--prop_radius_list -1 1 "
                             "--resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth";

    std::system(py_command.c_str());

    cv::FileStorage fs("output_task3/unimatch_output/rectified_left_disp_raw.yml", cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        printf("\nERROR: Could not load the raw disparity YML!\n");
        return -1;
    }

    Mat disp;
    fs["disparity"] >> disp;
    fs.release();
    
    printf("Successfully loaded Unimatch raw float disparity!\n");

    if(!point_cloud_filename.empty())
    {
        printf("storing the point cloud...");
        fflush(stdout);
        Mat xyz;
        
        reprojectImageTo3D(disp, xyz, Q, true);

        cv::FileStorage fs_xyz("output_task3/unimatch_output/xyz_data.yml", cv::FileStorage::WRITE);
        fs_xyz.write("xyz", xyz);
        fs_xyz.release();
        
        savePLY(point_cloud_filename.c_str(), xyz, img1);
        printf("\n");
    }

    return 0;
}
    