from keras.models import load_model
import matplotlib.pyplot as plt



model = load_model('C:\code\Rentgen2.h5') 


img_path = 'C:\Code\ormal-0381-0001.jpeg'  #Путь к проверяемому изображению

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size = (150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis=0)
img_tensor /= 255.

print('tensor shape: ', img_tensor.shape)



plt.imshow(img_tensor[0])
plt.show()

#запуск в режиме прогнозирования
preds = model.predict(img_tensor)
print('======================================\n')
print('Results: \n\n')
if preds<0.5:
    print("Отсутствие пневмонии, sure for {} %".format((100 - preds*200)))
else:
    print("Пневмония, sure for {} %".format((preds - 0.5)*200))
    









#first_layer_activation = activations[5]
#print(first_layer_activation.shape)

#import matplotlib.pyplot as plt
#plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')