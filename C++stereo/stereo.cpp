#include"stereo.hpp"


int Camera_function::calibrate(std::string left_txt, std::string right_txt,cv::Size chessboard,std::string xml_path,std::vector<int> params)
{
	if (left_txt.empty() || right_txt.empty())
	{
		std::cout << "txt文件为空" << std::endl;
		return 0;
	}
	else if (left_txt.rfind(".txt") == std::string::npos || right_txt.rfind(".txt") == std::string::npos)
	{
		std::wcout << "无txt文件可能输入文件错误" << std::endl;
		return 0;
	}
	if (xml_path.rfind(".xml") == std::string::npos)
	{
		std::cout << "xml文件路径错误" << std::endl;
		return 0;
	}
	if (params.empty())
	{
		params.assign(2, 0);
	}
	//左右路径的文件名
	std::string left_filename(left_txt), right_filename(right_txt);
	//左右路径的文件
	std::ifstream left_ifs(left_filename, std::ios::in);
	std::ifstream right_ifs(right_filename, std::ios::in);
	//获得棋盘格的尺寸
	int rows = chessboard.height;
	int cols = chessboard.width;
	int board_n = rows * cols;
	//左右图路径读取
	std::vector<std::string> left_path;
	std::vector<std::string> right_path;
	//世界点
	std::vector<std::vector<cv::Point3f>> objpoints;
	std::vector<cv::Point3f> points;
	//图像角点
	std::vector<std::vector<cv::Point2f>> left_points;
	std::vector<std::vector<cv::Point2f>> right_points;
	std::vector<cv::Point2f> left_corners;
	std::vector<cv::Point2f> right_corners;
	//图像路径
	std::string left_name;
	std::string right_name;
	//记录找到角点的图像个数
	int total = 0;
	//读取图像和灰度图
	cv::Mat left_img, right_img;
	cv::Mat left_gray, right_gray;
	//是否有角点
	bool left_ret, right_ret;
	//是否显示
	bool is_show = params[0];
	while (std::getline(left_ifs, left_name) && std::getline(right_ifs, right_name))
	{
		//存路径
		left_path.push_back(left_name);
		right_path.push_back(right_name);
		//读图片
		left_img = cv::imread(left_name);
		right_img = cv::imread(right_name);
		cv::cvtColor(left_img, left_gray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(right_img, right_gray, cv::COLOR_BGR2GRAY);

		//找角点
		left_ret = cv::findChessboardCorners(left_gray, chessboard, left_corners, cv::CALIB_CB_ADAPTIVE_THRESH);
		right_ret = cv::findChessboardCorners(right_gray, chessboard, right_corners, cv::CALIB_CB_ADAPTIVE_THRESH);
		if (!(left_ret & right_ret))
		{
			std::cout << "无角点" << std::endl;
			continue;
		}
		//利用粗角点寻找亚角点
		total++;
		cv::cornerSubPix(left_gray, left_corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 50, 1e-6));
		cv::cornerSubPix(right_gray, right_corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 50, 1e-6));
		if (is_show)
		{
			cv::drawChessboardCorners(left_img, chessboard, left_corners, left_ret);
			cv::drawChessboardCorners(right_img, chessboard, right_corners, right_ret);
			cv::imshow("left", left_img);
			cv::imshow("right", right_img);
			cv::waitKey(500);
		}
		//填入角点
		left_points.push_back(left_corners);
		right_points.push_back(right_corners);
	}

	//标志
	int flags =
		cv::CALIB_RATIONAL_MODEL + cv::CALIB_TILTED_MODEL;
	//图像大小
	cv::Size imagesize = left_img.size();
	//世界点填入
	for (int index = 0; index < total; index++)
	{
		points.clear();
		(points).swap(points);
		for (int i = 0; i < board_n; i++)
		{
			int x = i % cols;
			cv::Point3f  point(x, (i - x) / cols, 0);
			points.push_back(point);
		}
		objpoints.push_back(points);
	}
	//内参矩阵
	cv::Mat cameraMatrix1, cameraMatrix2;
	//畸变系数
	cv::Mat distcoeffs1(cv::Size(4, 1), CV_32FC1, cv::Scalar(0)), distcoeffs2(cv::Size(4, 1), CV_32FC1, cv::Scalar(0));
	//"开始标定"
	std::vector<cv::Mat> rvecs[2];
	std::vector<cv::Mat> tvecs[2];

	//计算内参和畸变系数
	double rms1 = cv::calibrateCamera(objpoints, left_points, imagesize, cameraMatrix1, distcoeffs1, rvecs[0], tvecs[0], 0, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
	double rms2 = cv::calibrateCamera(objpoints, right_points, imagesize, cameraMatrix2, distcoeffs2, rvecs[1], tvecs[1], 0, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
	//旋转矩阵和平移向量
	cv::Mat R, T;
	//基本矩阵和本征矩阵
	cv::Mat E, F;
	//计算R，T，E，E
	cv::stereoCalibrate(
		objpoints,
		left_points,
		right_points,
		cameraMatrix1,
		distcoeffs1,
		cameraMatrix2,
		distcoeffs2,
		imagesize,
		R,
		T,
		E,
		F,
		flags
	);

	//R1,R2:旋转矩阵
	//P1,P2:相机映射矩阵
	//Q:重投影矩阵
	cv::Mat R1, R2, P1, P2, Q;
	cv::stereoRectify(
		cameraMatrix1,
		distcoeffs1,
		cameraMatrix2,
		distcoeffs2,
		imagesize,
		R, T,
		R1, R2, P1, P2, Q
	);

	//关闭文件
	left_ifs.close();
	right_ifs.close();

	//是否保存数据
	bool is_save = params[1];
	if (is_save) {
		//保存数据
		cv::FileStorage file(xml_path, cv::FileStorage::WRITE);
		//写入数据
		file << "data" << "{"
			<< "camera1" << cameraMatrix1
			<< "dist1" << distcoeffs1
			<< "R1" << R1
			<< "P1" << P1
			<< "camera2" << cameraMatrix2
			<< "dist2" << distcoeffs2
			<< "R2" << R2
			<< "P2" << P2
			<< "Q" << Q
			<< "}";
		std::cout << "xml已保存" << std::endl;
		file.release();
	}
	return 1;
}







int Camera_function::fisheye_calibrate(std::string left_txt, std::string right_txt, cv::Size chessboard, std::string xml_path, std::vector<int> params)
{
	if (left_txt.empty()|| right_txt.empty())
	{
		std::cout << "txt文件为空" << std::endl;
		return 0;
	}
	else if (left_txt.rfind(".txt") != std::string::npos || right_txt.rfind(".txt") != std::string::npos)
	{
		std::wcout << "无txt文件可能输入文件错误" << std::endl;
		return 0;
	}
	if (xml_path.rfind(".xml") == std::string::npos)
	{
		std::cout << "xml文件路径错误" << std::endl;
		return 0;
	}
	if (params.empty())
	{
		params.assign(2, 0);
	}


	//左右路径的文件名
	std::string left_filename(left_txt), right_filename(right_txt);
	//左右路径的文件
	std::ifstream left_ifs(left_filename, std::ios::in);
	std::ifstream right_ifs(right_filename, std::ios::in);
	//获得棋盘格的尺寸
	int rows = chessboard.height;
	int cols = chessboard.width;
	int board_n = rows * cols;
	//左右图路径读取
	std::vector<std::string> left_path;
	std::vector<std::string> right_path;
	//世界点
	std::vector<std::vector<cv::Point3f>> objpoints;
	std::vector<cv::Point3f> points;
	//图像角点
	std::vector<std::vector<cv::Point2f>> left_points;
	std::vector<std::vector<cv::Point2f>> right_points;
	std::vector<cv::Point2f> left_corners;
	std::vector<cv::Point2f> right_corners;
	//图像路径
	std::string left_name;
	std::string right_name;
	//记录找到角点的图像个数
	int total = 0;
	//读取图像和灰度图
	cv::Mat left_img, right_img;
	cv::Mat left_gray, right_gray;
	//是否有角点
	bool left_ret, right_ret;
	//是否显示
	bool is_show = params[0];
	while (std::getline(left_ifs, left_name) && std::getline(right_ifs, right_name))
	{
		//存路径
		left_path.push_back(left_name);
		right_path.push_back(right_name);
		//读图片
		left_img = cv::imread(left_name);
		right_img = cv::imread(right_name);
		cv::cvtColor(left_img, left_gray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(right_img, right_gray, cv::COLOR_BGR2GRAY);
		//找角点
		left_ret = cv::findChessboardCorners(left_gray, chessboard, left_corners);
		right_ret = cv::findChessboardCorners(right_gray, chessboard, right_corners);
		if (!(left_ret & right_ret))
		{
			std::cout << "无角点" << std::endl;
			continue;
		}
		//利用粗角点寻找亚角点
		total++;
		cv::cornerSubPix(left_gray, left_corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 50, 1e-6));
		cv::cornerSubPix(right_gray, right_corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::MAX_ITER, 50, 1e-6));
		//显示
		if (is_show)
		{
			cv::drawChessboardCorners(left_img, chessboard, left_corners, left_ret);
			cv::drawChessboardCorners(right_img, chessboard, right_corners, right_ret);
			cv::resize(left_img, left_img, cv::Size(), 0.5, 0.5);
			cv::resize(right_img, right_img, cv::Size(), 0.5, 0.5);
			cv::imshow("left", left_img);
			cv::imshow("right", right_img);
			cv::waitKey(500);
		}
		//填入角点
		left_points.push_back(left_corners);
		right_points.push_back(right_corners);
	}
	//标志
	int flags = cv::CALIB_FIX_ASPECT_RATIO +
		cv::CALIB_RATIONAL_MODEL + cv::CALIB_TILTED_MODEL;
	//图像大小
	cv::Size imagesize = left_img.size();
	//世界点填入
	for (int index = 0; index < total; index++)
	{
		points.clear();
		(points).swap(points);
		for (int i = 0; i < board_n; i++)
		{
			int x = i % cols;
			cv::Point3f  point(x, (i - x) / cols, 0);
			points.push_back(point);
		}
		objpoints.push_back(points);
	}


	//内参矩阵
	cv::Mat K1, K2;
	//畸变系数
	cv::Mat D1, D2;
	//"开始标定"
	std::vector<cv::Mat> rvecs[2];
	std::vector<cv::Mat> tvecs[2];




	//计算内参和畸变系数
	double rms1 = cv::fisheye::calibrate(objpoints, left_points, imagesize, K1, D1, rvecs[0], tvecs[0], 0, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
	double rms2 = cv::fisheye::calibrate(objpoints, right_points, imagesize, K2, D2, rvecs[1], tvecs[1], 0, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
	//旋转矩阵和平移向量
	cv::Mat R, T;
	//基本矩阵和本征矩阵
	cv::Mat E, F;
	//计算R，T
	cv::fisheye::stereoCalibrate(
		objpoints,
		left_points,
		right_points,
		K1,
		D1,
		K2,
		D2,
		imagesize,
		R,
		T,
		flags
	);

	//R1,R2:旋转矩阵
	//P1,P2:相机映射矩阵
	//Q:重投影矩阵
	cv::Mat R1, R2, P1, P2, Q;
	cv::fisheye::stereoRectify(
		K1,
		D1,
		K2,
		D2,
		imagesize,
		R, T,
		R1, R2, P1, P2, Q,
		0
	);

	//关闭文件
	left_ifs.close();
	right_ifs.close();

	//是否保存数据
	bool is_save = params[1];
	if (is_save) {
		//保存数据
		cv::FileStorage file(xml_path, cv::FileStorage::WRITE);
		//写入数据
		file << "data" << "{"
			<< "K1" << K1
			<< "D1" << D1
			<< "R1" << R1
			<< "P1" << P1
			<< "K2" << K2
			<< "D2" << D2
			<< "R2" << R2
			<< "P2" << P2
			<< "Q" << Q
			<< "}";
		file.release();
	}
	return 1;
}
