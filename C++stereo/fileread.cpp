
#include"fileread.hpp"

int getFiles(std::string path, std::vector<std::string>& files){
    //文件句柄  
    intptr_t hFile = 0;
    //文件信息，声明一个存储文件信息的结构体  
    struct _finddata_t fileinfo;
    std::string p;  //字符串，存放路径
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)//若查找成功，则进入
    {
        do
        {
            //如果是目录,迭代之（即文件夹内还有文件夹）  
            if ((fileinfo.attrib & _A_SUBDIR))
            {
                //文件名不等于"."&&文件名不等于".."
                //.表示当前目录
                //..表示当前目录的父目录
                //判断时，两者都要忽略，不然就无限递归跳不出去了！
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
            }
            //如果不是,加入列表  
            else
            {
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        //_findclose函数结束查找
        _findclose(hFile);
    }
    else
    {
        std::cerr << "无文件" << std::endl;
        return 0;
    }
    return 1;
}


int txtwrite(std::string filepath, std::vector<std::string>& files)
{
    if (filepath.empty())
    {
        std::cerr << "路径为空" << std::endl;
        return 0;
    }
    else if (filepath.find(".txt")==std::string::npos)
    {
        std::cerr << "非txt文件路径" << std::endl;
        return 0;
    }
    std::ofstream ofs(filepath, std::ios::out);
    //迭代器迭代
    for (auto it = files.begin(); it != files.end(); it++)
    {
        //写入文件名称
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