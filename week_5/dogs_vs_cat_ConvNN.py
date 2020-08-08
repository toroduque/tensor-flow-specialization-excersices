import os

# Organize data from folders
base_dir = './tmp/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Print the total number of  cat and images in the `train` and `validation` dirs
print('total training cat images : ', len(os.listdir(train_cats_dir)))
print('total training dog images : ', len(os.listdir(train_dogs_dir)))

print('total validation cats images : ', len(os.listdir(validation_cats_dir)))
print('total cvalidation dogs images : ', len(os.listdir(validation_dogs_dir)))


# Build small model to get ~72% accuracy

import tensorflow as tf

model = tf.keras.models.Sequential([
    # 1st Conv layer
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    # 2nd Conv layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #  3rd Conv layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    # Flatten the reults to feed into a Deep NN
    tf.keras.layers.Flatten(),
    
    # 512 neurons hidden layer
    tf.keras.layers.Dense(512, activation='relu'),

    # Output only one. Using Sigmoid to return either 0 or 1. (0 = cats / 1 = dogs)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# Compile the Model
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr = 0.001),
              loss='binary_crossentropy',
              metrics=['acc'])


# Data Processing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescale by 1./255
train_datagen = ImageDataGenerator( rescale = 1./255 )
test_datagen = ImageDataGenerator( rescale = 1./255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    batch_size=20,
    class_mode='binary',
    target_size=(150, 150)
)

# Training
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=15,
                    validation_steps=50,
                    verbose=2)

print(history)