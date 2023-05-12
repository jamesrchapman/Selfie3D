from keras.layers import Input, Conv3D, MaxPooling3D, Dense, Flatten, Dropout, Activation, Permute, Reshape, Lambda, Dot
from keras.models import Model

# Define the input shape
input_shape = (16, 112, 112, 3)  # 16 frames of 112x112 RGB images

# Define the CNN layers
input_layer = Input(shape=input_shape)
conv_layer1 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding='same')(input_layer)
pool_layer1 = MaxPooling3D(pool_size=(1, 2, 2))(conv_layer1)
conv_layer2 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same')(pool_layer1)
pool_layer2 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer2)
conv_layer3 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same')(pool_layer2)
pool_layer3 = MaxPooling3D(pool_size=(2, 2, 2))(conv_layer3)
flatten_layer = Flatten()(pool_layer3)

# Define the self-attention layer
attention_probs = Dense(64, activation='softmax')(flatten_layer)
attention_mul = Lambda(lambda x: x[0]*x[1])([flatten_layer, attention_probs])
reshape_layer = Reshape((64, -1))(attention_mul)
permute_layer = Permute((2, 1))(reshape_layer)
attention_scores = Dense(64, activation='softmax')(permute_layer)
attention_vec = Lambda(lambda x: x[0]*x[1])([permute_layer, attention_scores])
attention_vec = Permute((2, 1))(attention_vec)
attention_vec = Flatten()(attention_vec)

# Define the fully connected layers
dense_layer1 = Dense(256, activation='relu')(attention_vec)
dropout_layer1 = Dropout(0.5)(dense_layer1)
dense_layer2 = Dense(128, activation='relu')(dropout_layer1)
dropout_layer2 = Dropout(0.5)(dense_layer2)
output_layer = Dense(num_classes, activation='softmax')(dropout_layer2)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
