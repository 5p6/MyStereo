#ifndef FILEREAD_HPP
#include<iostream>
#include<io.h>
#include<vector>
#include<string>
#include<fstream>
/*@brief ��ȡ�ļ����������ļ������� ���ҷ�������·�� 
* @param path �ļ�·��
* @param files װ���ļ�·��������
*/
int getFiles(std::string path, std::vector<std::string>& files);
/*@brief ��װ���ļ�·�����������ַ�д�뵽txt�ļ���
* @param filepath txt�ļ�·��
* @param fiels  ������txt�ļ�����ļ�·������ 
*/
int txtwrite(std::string filepath, std::vector<std::string>& files);



/*@brief ��ͼ���ļ����е�ͼ��·�����浽һ��txt�ļ���
* @param image_files ͼ���ļ����ļ���·��
* @param txtfiles txt�ļ���·��
*/
int image2txt(std::string& image_fils, std::string& txtfiles);

#endif // !FILEREAD_HPP
