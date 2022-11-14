
#include"fileread.hpp"

int getFiles(std::string path, std::vector<std::string>& files){
    //�ļ����  
    intptr_t hFile = 0;
    //�ļ���Ϣ������һ���洢�ļ���Ϣ�Ľṹ��  
    struct _finddata_t fileinfo;
    std::string p;  //�ַ��������·��
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)//�����ҳɹ��������
    {
        do
        {
            //�����Ŀ¼,����֮�����ļ����ڻ����ļ��У�  
            if ((fileinfo.attrib & _A_SUBDIR))
            {
                //�ļ���������"."&&�ļ���������".."
                //.��ʾ��ǰĿ¼
                //..��ʾ��ǰĿ¼�ĸ�Ŀ¼
                //�ж�ʱ�����߶�Ҫ���ԣ���Ȼ�����޵ݹ�������ȥ�ˣ�
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
            }
            //�������,�����б�  
            else
            {
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        //_findclose������������
        _findclose(hFile);
    }
    else
    {
        std::cerr << "���ļ�" << std::endl;
        return 0;
    }
    return 1;
}


int txtwrite(std::string filepath, std::vector<std::string>& files)
{
    if (filepath.empty())
    {
        std::cerr << "·��Ϊ��" << std::endl;
        return 0;
    }
    else if (filepath.find(".txt")==std::string::npos)
    {
        std::cerr << "��txt�ļ�·��" << std::endl;
        return 0;
    }
    std::ofstream ofs(filepath, std::ios::out);
    //����������
    for (auto it = files.begin(); it != files.end(); it++)
    {
        //д���ļ�����
        ofs << *it << "\n";
    }
    ofs.close();
    return 1;
}


int image2txt(std::string& image_fils, std::string& txtfiles)
{

    std::vector<std::string> files;
    getFiles(image_fils, files);
    txtwrite(txtfiles, files);
    return 1;
}