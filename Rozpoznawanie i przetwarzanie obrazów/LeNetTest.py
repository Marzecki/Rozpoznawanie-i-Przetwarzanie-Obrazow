from keras.models import Sequential
import keras.layers.convolutional
import keras.layers.core
import keras.backend
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


#labels = ['20', '30', '50', '60', '70', '80', 'not80', '100', '120', 'noPassing']

#Preprocesing:
#rotation_range - możliwy obrót w stopniach
#width/height_shift_range - możliwa translacja obrazu w 'procentach (0.2 = 20%)
#rescale - mapowanie do zakresu 0-1
#shear_range - możliwe odchylenie w przestrzeni 3-D
#zoom_range - możliwe przybliżenie
#fill_mode - jak wypełnić puste piksele ktore powstaly w wyniku transformacji
#validation_split - ile pojdzie na walidacje
train_dataGen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    validation_split=0.2
)

#pobranie obrazów do trenowania i walidacji (w tym zamienienie na grayscale)
train_generator = train_dataGen.flow_from_directory(
    'Dataset/', 
    target_size=(32, 32),
    class_mode='categorical', 
    batch_size=64, 
    color_mode='grayscale',
    subset='training'
)

print(train_generator.image_shape)

validation_generator = train_dataGen.flow_from_directory(
    'Dataset/',
    target_size=(32, 32),
    class_mode='categorical',                                                        
    batch_size=64,
    color_mode='grayscale',
    subset='validation'
)

print(validation_generator.image_shape)

print('\n\n\n\n')



#Model
model = keras.Sequential()

model.add(keras.layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32,32,1)))     # pierwsza warstwa
model.add(keras.layers.AveragePooling2D())                                                                  # druga warstwa 

model.add(keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))                           # trzecia warstwa
model.add(keras.layers.AveragePooling2D())                                                                  # czwarta warstwa

model.add(keras.layers.Flatten())                                                                          

model.add(keras.layers.Dense(units=120, activation='relu'))                                                 # piata warstwa

model.add(keras.layers.Dense(units=84, activation='relu'))                                                  # szosta warstwa

model.add(keras.layers.Dense(units=10, activation = 'softmax'))                                             # siodma warstwa

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

print(model.summary())


#trenowanie
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // 64,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // 64,
    epochs = 3)


############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score =model.evaluate(train_generator,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])
