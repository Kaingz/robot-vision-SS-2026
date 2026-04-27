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

    Vec3f point3D_1 = xyz.at<Vec3f>(pt1.y, pt1.x);
    Vec3f point3D_2 = xyz.at<Vec3f>(pt2.y, pt2.x);

    double distance = cv::norm(point3D_1 - point3D_2) / 1000.0;
    double delta_X = abs(point3D_1[0] - point3D_2[0]) / 1000.0;
    double delta_Y = abs(point3D_1[1] - point3D_2[1]) / 1000.0;
    double delta_Z = abs(point3D_1[2] - point3D_2[2]) / 1000.0;


    printf("Point 1 coordinates: X:%.3f, Y:%.3f, Z:%.3f\n", point3D_1[0]/1000.0, point3D_1[1]/1000.0, point3D_1[2]/1000.0);
    printf("Point 2 coordinates: X:%.3f, Y:%.3f, Z:%.3f\n", point3D_2[0]/1000.0, point3D_2[1]/1000.0, point3D_2[2]/1000.0);
    printf("---------------------------\n");
    printf("Delta X:  %.4f m\n", delta_X);
    printf("Delta Y:  %.4f m\n", delta_Y);
    printf("Delta Z:  %.4f m\n", delta_Z);
    printf("---------------------------\n");
    printf("Euclidean distance: %.4f m\n\n", distance);

    Mat measurement_img = img.clone();
    circle(measurement_img, pt1, 7, Scalar(0, 0, 255), -1);
    circle(measurement_img, pt2, 7, Scalar(0, 0, 255), -1);
    line(measurement_img, pt1, pt2, Scalar(0, 255, 0), 3);

    imwrite("report_measurement_3.png", measurement_img);

    return 0;
}
