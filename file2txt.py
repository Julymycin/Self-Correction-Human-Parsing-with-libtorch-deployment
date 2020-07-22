import os

data_path='/home/qiu/Downloads/datasets/ICCV15_fashion_dataset(ATR)/humanparsing/train_images'
txt_path='/home/qiu/Downloads/datasets/ICCV15_fashion_dataset(ATR)/humanparsing/train_id.txt'
filelist=os.listdir(data_path)
with open(txt_path,'w') as fp:
    for file in filelist:
        if file.endswith('.jpg'):
            name=file.split('.')[0]
            fp.write(name+'\n')
