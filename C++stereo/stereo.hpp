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

	/*@brief ��ͨ˫Ŀ�궨����
	* @param left_txt ����������궨ͼƬ·����txt�ļ���txt�ļ��У����е���ͼ·��������
	* @param right_txt ����������궨ͼƬ·����txt�ļ���txt�ļ��У����е���ͼ·��������
	* @param chessboard ���̸�Ľǵ�ߴ磬(x,y)�ʹ���ÿһ����x���ǵ�ÿ����y���ǵ�
	* @param xml_path ������ݱ����xml�ļ�·��
	* @param params �ڲ�������������һλ�Ƿ���ʾ���ҵ��ǵ��ͼ�񣬵ڶ�λ���Ƿ񱣴����е�����Ϊxml�ļ�
	*/
	static int calibrate(std::string left_txt, std::string right_txt, cv::Size chessboard, std::string xml_path, std::vector<int> params);









	/*@brief ��ͨ˫Ŀ�궨����
	* @param left_txt ����������궨ͼƬ·����txt�ļ���txt�ļ��У����е���ͼ·��������
	* @param right_txt ����������궨ͼƬ·����txt�ļ���txt�ļ��У����е���ͼ·��������
	* @param chessboard ���̸�Ľǵ�ߴ磬(x,y)�ʹ���ÿһ����x���ǵ�ÿ����y���ǵ�
	* @param xml_path ������ݱ����xml�ļ�·��
	* @param params �ڲ�������������һλ�Ƿ���ʾ���ҵ��ǵ��ͼ�񣬵ڶ�λ���Ƿ񱣴����е�����Ϊxml�ļ�
	*/
	static int fisheye_calibrate(std::string left_txt, std::string right_txt, cv::Size chessboard, std::string xml_path, std::vector<int> params);



};



#endif // !STEREO_HPP
