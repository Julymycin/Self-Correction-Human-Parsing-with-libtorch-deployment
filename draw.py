import matplotlib.pyplot as plt
import matplotlib.image as mpimg

file_path = '/home/qiu/Projects/lidingtiao/full_cut/imgs/'
atrpath = './output_atr/'
lippath = './output_lip/'
pascalpath = './output_pascal/'
img_name = '6_29_542_0.png'
choices = ['lip', 'atr', 'pascal']
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
src = mpimg.imread(file_path + img_name)
plt.imshow(src)
plt.subplot(2, 2, 2)
atri = mpimg.imread(atrpath + img_name)
plt.imshow(atri)
plt.subplot(2, 2, 3)
lipi = mpimg.imread(lippath + img_name)
plt.imshow(lipi)
plt.subplot(2, 2, 4)
pascali = mpimg.imread(pascalpath + img_name)
plt.imshow(pascali)
plt.show()
