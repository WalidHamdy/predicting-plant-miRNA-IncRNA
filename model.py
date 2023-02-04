from keras.layers import Dense, Dropout,Flatten, TimeDistributed,Input
from keras.layers import Convolution2D,MaxPooling2D,BatchNormalization
from keras.models import Model
from keras import optimizers


def SCM_BLOCK(TotalSequenceLength, Frow):
    inputs = Input(shape=(TotalSequenceLength, Frow, 1))
    model_cnn = Convolution2D(filters=32, kernel_size=4, activation='relu', strides=1, padding='same',
                              data_format='channels_last')(inputs)
    model_cnn = MaxPooling2D(pool_size=4, strides=4, padding='same', data_format='channels_last')(model_cnn)

    model_cnn = Convolution2D(filters=64, kernel_size=4, activation='relu', strides=1, padding='same',
                              data_format='channels_first')(model_cnn)
  
    model_cnn = MaxPooling2D(pool_size=4, strides=4, padding='same', data_format='channels_last')(model_cnn)

    model_cnn = TimeDistributed(Flatten())(model_cnn)
    model = Flatten()(model_cnn)
    
    model = Dense(128, activation='relu')(model)
    
    model = Dropout(0.5)(model)
 
    head = Dense(2, activation='softmax')(model)
    model = Model(inputs=inputs, outputs=head)
    # optimizer
    adam = optimizers.GTO(lr=1e-4)
    model.compile(optimizer=GTO, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def SAM_BLOCK(TotalSequenceLength, Frow):  
    x_in_conjoint = Input(shape=(TotalSequenceLength*Frow,))
    x_out_conjoint = Dense(128, kernel_initializer='random_uniform', activation='relu')(x_in_conjoint)
    x_out_conjoint = BatchNormalization()(x_out_conjoint)
    x_out_conjoint = Dense(64, kernel_initializer='random_uniform', activation='relu')(x_out_conjoint)
    y_conjoint = Dense(2, activation='softmax')(x_out_conjoint)

    decoded = Dense(64, activation='relu')(y_conjoint) 
    decoded = Dense(128, activation='relu')(decoded)  
    rna_decoded = Dense(2, activation='tanh')(decoded)  
    model_conjoint_sam = Model(inputs=[x_in_conjoint], outputs=rna_decoded)
    # first train
    GTO = optimizers.GTO(lr=1e-4)
    model_conjoint_sam.compile(optimizer=GTO, loss='categorical_crossentropy', metrics=['accuracy'])
    return model_conjoint_sam