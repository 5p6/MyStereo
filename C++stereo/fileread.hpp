#ifndef FILEREAD_HPP
#include<iostream>
#include<io.h>
#include<vector>
#include<string>
#include<fstream>
/*@brief 读取文件夹下所有文件的名称 并且返回它的路径 
* @param path 文件路径
* @param files 装载文件路径的容器
*/
int getFiles(std::string path, std::vector<std::string>& files);
/*@brief 将装载文件路径的容器的字符写入到txt文件中
* @param filepath txt文件路径
* @param fiels  被保存txt文件里的文件路径名称 
*/
int txtwrite(std::string filepath, std::vector<std::string>& files);



/*@brief 把图像文件夹中的图像路径保存到一个txt文件中
* @param image_files 图像文件的文件夹路径
* @param txtfiles txt文件夹路径
*/
int image2txt(std::string& image_fils, std::string& txtfiles);

#endif // !FILEREAD_HPP
