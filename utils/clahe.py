import cv2  
import numpy as np  
import os
from PIL import Image, ImageEnhance  
import os  
import numpy as np  

gamma = 1.0# 例如，设置为0.5将增加亮度 
def apply_clahe_to_brightness(image, clahe):  
    # 将图像从BGR转换为HSV  
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  
      
    # 获取亮度通道V
    # r_channel = image[:, :, 0]
    # g_channel = image[:, :, 1]
    # b_channel = image[:, :, 2]  
      
    # # 应用CLAHE到亮度通道  
    # clahe_r = clahe.apply(r_channel)
    # clahe_g = clahe.apply(g_channel)
    # clahe_b = clahe.apply(b_channel) 
    image = clahe.apply(image)

      
    # 将CLAHE处理后的亮度通道与原始HSV的H和S通道合并  
    # image[:, :, 0] = clahe_r 
    # image[:, :, 1] = clahe_g 
    # image[:, :, 2] = clahe_b
      
    # # 将HSV转换回BGR  
    # clahe_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  
      
    return image  
  
def process_images_with_clahe(input_dir, output_dir):  
    # 创建一个CLAHE对象  
    clahe = cv2.createCLAHE(clipLimit=15, tileGridSize=(1, 1))  
      
    # 遍历输入文件夹中的所有文件  
    for filename in os.listdir(input_dir):  
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 可以根据需要添加更多格式  
            # 构建输入和输出文件的完整路径  
            input_path = os.path.join(input_dir, filename)  
            output_path = os.path.join(output_dir, filename)  
              
            # 读取图像  
            image = cv2.imread(input_path)  
            
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  
        
            # 应用直方图均衡化  
            h, s, v = cv2.split(hsv_image)
            # 应用CLAHE到亮度通道  
            v_equalized = clahe.apply(v)
            # 合并通道  
            enhanced_image = cv2.merge([h, s, v_equalized])   
            image = cv2.cvtColor(enhanced_image, cv2.COLOR_HSV2BGR)
            
            # 将图像数组转换为浮点型以进行数学运算  
            img_array = image.astype('float')  
            # 应用gamma校正  
            img_array_gamma_corrected = np.power(img_array / 255.0, gamma) * 255.0  
            # 限制像素值范围在0-255之间  
            img_array_gamma_corrected = np.clip(img_array_gamma_corrected, 0, 255)  
            # 将浮点型数组转回uint8类型  
            image = img_array_gamma_corrected.astype('uint8')  
            # 将NumPy数组转回PIL图像  
            # image = Image.fromarray(img_array_gamma_corrected, 'RGB')
            
             
            if image is not None:  
                cv2.imwrite(output_path, image)  
                print(f"Processed and saved: {filename}")  
            else:  
                print(f"Error reading: {filename}")   
  
# 定义输入和输出文件夹  
input_dir = '/mnt/qzf/Lush-NeRF/data/0060_clahe/images'  # 替换为你的输入文件夹路径  
output_dir = '/mnt/qzf/Lush-NeRF/data/0060-test/images'  # 替换为你的输出文件夹路径  
  
# 确保输出文件夹存在  
if not os.path.exists(output_dir):  
    os.makedirs(output_dir)  
  
# 处理文件夹中的所有图像  
process_images_with_clahe(input_dir, output_dir)