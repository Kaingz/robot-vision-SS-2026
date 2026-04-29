#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    Mat img = imread("output_task3/left/rectified_left.png", IMREAD_COLOR);

    Mat xyz;
    FileStorage fs("output_task3/unimatch_output/xyz_data.yml", FileStorage::READ);
    fs["xyz"] >> xyz;
    fs.release();

    Point2i pt1(1364, 816);
    Point2i pt2(1847, 703); 

    Vec3f pt3d_1 = xyz.at<Vec3f>(pt1.y, pt1.x);
    Vec3f pt3d_2 = xyz.at<Vec3f>(pt2.y, pt2.x);

    double distance = cv::norm(pt3d_1 - pt3d_2) / 1000.0;
    printf("Euclidean distance: %.4f m\n", distance);

    Mat measurement_img = img.clone();
    circle(measurement_img, pt1, 10, Scalar(0, 0, 255), -1);
    circle(measurement_img, pt2, 10, Scalar(0, 0, 255), -1);
    line(measurement_img, pt1, pt2, Scalar(0, 255, 0), 6);

    imwrite("report_measurement_3.png", measurement_img);

    return 0;
}
