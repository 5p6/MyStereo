import torch, gc
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0)) # 查看使用的设备名称


a = torch.ones((3,1))
a = a.cuda(0)
b = torch.ones((3,1)).cuda(0)
print(a+b)
gc.collect()
torch.cuda.empty_cache()
