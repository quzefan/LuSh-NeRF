import os  
from PIL import Image  
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim  
import lpips  
import torch  
import numpy as np  
import glob
  
# 初始化LPIPS损失函数  
net = lpips.LPIPS(net='alex')  
net.eval()  
  
# 文件夹路径  
folder1 = '..'  # 第一个文件夹路径  
folder2 = '..'  # 第二个文件夹路径  
  
# 确保两个文件夹中的图像数量相同  
# images1 = sorted(glob.glob(os.path.join(folder1, '[0-9][0-9][0-9].png')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))  
images1 = sorted(glob.glob(os.path.join(folder1, '[0-9][0-9][0-9].png')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0])) 
images2 = sorted(glob.glob(os.path.join(folder2, '[0-9][0-9][0-9][0-9].png')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))
  
if len(images1) != len(images2):  
    raise ValueError("The number of images in the two folders is not the same.")  
  
# 对两个文件夹中的图像进行排序（如果需要）以确保比较的是相同的图像  
# 这里假设你已经按照某种方式（例如文件名）排序了它们  
  
number, k = 0, 0
PSNR, SSIM, LPIPS = 0, 0, 0
# 计算PSNR, SSIM, 和LPIPS  
for img1_path, img2_path in zip(images1, images2):  
    # 加载图像  
    img1 = Image.open(img1_path).convert('RGB')  
    img2 = Image.open(img2_path).convert('RGB')  
  
    # 确保图像具有相同的尺寸  
    if img1.size != img2.size:  
        img2 = img2.resize(img1.size)  
  
    # 转换为张量（对于LPIPS）  
    img1_tensor = torch.from_numpy(np.array(img1).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0  
    img2_tensor = torch.from_numpy(np.array(img2).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0  
    
    # 加载图像并转换为numpy数组  
    img1_array = np.array(img1)  
    img2_array = np.array(img2)
    
    min_size = min(img1_array.shape[0], img1_array.shape[1])  # 假设图像是二维的  
    win_size = min(11, min_size - 1) if min_size > 11 else 11  # 确保 win_size 是奇数且小于等于 min_size  
    # 计算PSNR  
    psnr_value = compare_psnr(img1_array, img2_array, data_range=255)  
  
    # 计算SSIM  
    ssim_value = compare_ssim(img1_array, img2_array, win_size=win_size, multichannel=True, channel_axis=-1) 
  
    # 计算LPIPS  
    lpips_value = net(img1_tensor, img2_tensor).item()  
    
    if number % 8 == 0:
        PSNR+=psnr_value
        SSIM+=ssim_value
        LPIPS+=lpips_value
        k+=1
    number += 1
print(f"PSNR: {PSNR / k:.2f}, SSIM: {SSIM / k:.4f}, LPIPS: {LPIPS / k:.4f}")