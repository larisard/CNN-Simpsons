import os
import tempfile
import zipfile
import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import InputLayer, Conv2D, Dense, Dropout, MaxPooling2D, Flatten #type:ignore
from tensorflow.keras.preprocessing import image #type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #remover as mensagens do tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

arquivoTemp = tempfile.TemporaryDirectory()

with zipfile.ZipFile('dataset_personagens.zip', 'r') as zip:
     zip.extractall(arquivoTemp.name)
     
redeNeural = Sequential()

redeNeural.add(InputLayer(shape = (48, 48, 3)))
redeNeural.add(Conv2D(filters = 32, activation = 'relu', kernel_size = (3,3)))
redeNeural.add(MaxPooling2D(pool_size = (2,2)))

redeNeural.add(Conv2D(filters = 32, activation = 'relu', kernel_size = (3,3)))
redeNeural.add(MaxPooling2D(pool_size = (2,2)))

redeNeural.add(Conv2D(filters = 32, activation = 'relu', kernel_size = (3,3)))
redeNeural.add(MaxPooling2D(pool_size = (2,2)))

redeNeural.add(Flatten())

redeNeural.add(Dense(units = 32, activation = 'relu'))
redeNeural.add(Dropout(0.2))

redeNeural.add(Dense(units = 1, activation = 'sigmoide'))

redeNeural.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

redeNeural.summarize()


geradorTreinamento = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 7,
                                        horizontal_flip = True,
                                        shear_range = 0.2,
                                        height_shift_range = 0.07,
                                        zoom_range = 0.2)

geradorTeste = ImageDataGenerator(rescale = 1./255)

imagensTreinamento = geradorTreinamento.flow_from_directory(f'{arquivoTemp.name}/dataset_personagens/training_set', target_size = (48,48),
                                                            batch_size = 32, class_mode = 'binary')

imagensTeste = geradorTeste.flow_from_directory(f'{arquivoTemp.name}/dataset_personagens/test_set', target_size = (48,48),
                                                            batch_size = 32, class_mode = 'binary')

redeNeural.fit_generator(imagensTreinamento, steps_per_epoch = 196, epochs = 100,
                         validation_data = imagensTeste, validation_steps = 73)


     

    