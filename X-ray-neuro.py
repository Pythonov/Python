
"""
@author: Pimonov EI
"""

#архитектура сети

import os
from keras import layers,models,optimizers


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3))) #150,150

model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #слой прореживания
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()  #вывод архитектуры построенной сети


#Компиляция

model.compile(loss = 'binary_crossentropy', #функция потерь
              optimizer=optimizers.RMSprop(lr=1e-4), #функция оптимизатора
              metrics=['acc'])

base_dir = 'C:\code\X-RAY'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


# преобразование набора данных для обучения
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150), #150,150
    batch_size=32,
    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150), #150,150
    batch_size=16,
    class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape: ', data_batch.shape)
    print('labels batch shape: ', labels_batch.shape)
    break



#обучение модели с использованием генератора пакетов
history = model.fit_generator(
    train_generator,
    steps_per_epoch=83,
    epochs=45, #(35 for 200,200  40 for 150,150)
    validation_data = validation_generator,
    validation_steps = 8)

#сохранение файла модели
model.save('Rentgen4.h5')


#построение графиков точности и потерь
import matplotlib.pyplot as plt


def smoothie (points,strength=0.8): #Сглаживание графиков
    smoothie_points = []
    for point in points:
        if smoothie_points:
            previous = smoothie_points[-1]
            smoothie_points.append(previous*strength+point*(1-strength))
        else:
            smoothie_points.append(point)
    return smoothie_points


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)


plt.plot(epochs, smoothie(acc), 'bo', label='Training acc')
plt.plot(epochs, smoothie(val_acc), 'b', label='Validation acc')
plt.title('X-Ray Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, smoothie(loss), 'bo', label='Training loss')
plt.plot(epochs, smoothie(val_loss), 'b', label='Validation loss')
plt.title('X-Ray Training and validation loss')
plt.legend()
plt.show()

