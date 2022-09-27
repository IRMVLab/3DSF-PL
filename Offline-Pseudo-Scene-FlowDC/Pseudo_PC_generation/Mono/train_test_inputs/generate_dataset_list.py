import os
 
#返回原始图像路径名称
def img_file_name(file_dir):   
    L=''
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if file == 'img.png':  
                L = os.path.join(root, file)
#                print(L)
#                file_name = file[0:-4]  #去掉.png后缀
#                L.append(file_name)  
#                L.append(' '+'this is anohter file\'s name')
    return L 
 
#返回标注图像路径名称
def label_file_name(file_dir):   
    L=''  
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if file == 'label.png':  
                L = os.path.join(root, file) 
#                file_name = file[0:-4]  #去掉.png后缀
#                L.append(file_name)  
#                L.append(' '+'this is anohter file\'s name')
    return L  
 
imgdir = '/dataset/SeasonDepth_testset/SeasonDepth_testset/images'
list_txt_file = '../train_test_inputs/Season_test_c0list.txt'

os.makedirs(os.path.dirname(list_txt_file), exist_ok=True)
docs = os.listdir(imgdir) #找出文件夹下所有的文件

for root, dirs, files in os.walk(imgdir):
    for f in files:
        rootpathss = os.path.join(root, f)
        imagess = rootpathss[49:]
        if 'c1' == rootpathss[62:64]:
            rootpathss = rootpathss[56:]
            depthmap = 'depth/' + rootpathss[:-3] +  'png'
            txt_name = imagess + ' ' + depthmap + ' ' + '873.3826'
            with open(list_txt_file, 'a') as f:
                f.write(txt_name+'\n')
            f.close()
'''for name in docs:
    print('os.walk:',os.walk(imgdir))
    if name.endswith("_json"): #找到每个_json结尾的文件夹
        print(name)
        label_folder = imgdir+'/'+name
        txt_name = img_file_name(label_folder)+' '+label_file_name(label_folder)
        with open(list_txt_file, 'a') as f:
            f.write(txt_name+'\n')
        f.close()'''