#ifndef STEREO_HPP
#define STEREO_HPP

#include<deque>
#include<opencv2/calib3d.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/photo.hpp>
#include<opencv2/imgproc.hpp>
#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include<opencv2/dnn.hpp>
#include<io.h>

class Camera_function
{
public:

	/*@brief 普通双目标定程序
	* @param left_txt 关于左相机标定图片路径的txt文件，txt文件中，所有的左图路径都在内
	* @param right_txt 关于右相机标定图片路径的txt文件，txt文件中，所有的右图路径都在内
	* @param chessboard 棋盘格的角点尺寸，(x,y)就代表每一列有x个角点每行有y个角点
	* @param xml_path 相关数据保存的xml文件路径
	* @param params 内部参数调整，第一位是否显示已找到角点的图像，第二位是是否保存所有的数据为xml文件
	*/
	static int calibrate(std::string left_txt, std::string right_txt, cv::Size chessboard, std::string xml_path, std::vector<int> params);









	/*@brief 普通双目标定程序
	* @param left_txt 关于左相机标定图片路径的txt文件，txt文件中，所有的左图路径都在内
	* @param right_txt 关于右相机标定图片路径的txt文件，txt文件中，所有的右图路径都在内
	* @param chessboard 棋盘格的角点尺寸，(x,y)就代表每一列有x个角点每行有y个角点
	* @param xml_path 相关数据保存的xml文件路径
	* @param params 内部参数调整，第一位是否显示已找到角点的图像，第二位是是否保存所有的数据为xml文件
	*/
	static int fisheye_calibrate(std::string left_txt, std::string right_txt, cv::Size chessboard, std::string xml_path, std::vector<int> params);



};



#endif // !STEREO_HPP
