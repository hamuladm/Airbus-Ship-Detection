
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from keras import models, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from metrics import FocalLoss, dice_coef
from utils import batch_data

from config import *

masks = pd.read_csv(os.path.join(BASE_DIR, 'train_ship_segmentations_v2.csv'))

masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
masks.drop(['ships'], axis=1, inplace=True)

balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

train_ships, valid_ships = train_test_split(balanced_train_df, test_size=0.2, stratify=balanced_train_df.ships)

train_df = pd.merge(masks, train_ships)
valid_df = pd.merge(masks, valid_ships)


def UnetCNN(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    C1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    C1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C1)
    P1 = layers.MaxPooling2D((2, 2))(C1)

    C2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(P1)
    C2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C2)
    P2 = layers.MaxPooling2D((2, 2))(C2)

    C3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(P2)
    C3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C3)
    P3 = layers.MaxPooling2D((2, 2))(C3)

    C4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(P3)
    C4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C4)
    P4 = layers.MaxPooling2D(pool_size=(2, 2))(C4)

    C5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(P4)
    C5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C5)

    U6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(C5)
    U6 = layers.concatenate([U6, C4])
    C6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U6)
    C6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C6)

    U7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(C6)
    U7 = layers.concatenate([U7, C3])
    C7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U7)
    C7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C7)

    U8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(C7)
    U8 = layers.concatenate([U8, C2])
    C8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U8)
    C8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C8)

    U9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(C8)
    U9 = layers.concatenate([U9, C1], axis=3)
    C9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U9)
    C9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C9)

    D = layers.Conv2D(1, (1, 1), activation='sigmoid')(C9)

    seg_model = models.Model(inputs=[inputs], outputs=[D])

    return seg_model


built_model = UnetCNN()



train_dfrm = batch_data(BATCH_SIZE, train_df['ImageId'], masks)
valid_dfrm = batch_data(BATCH_SIZE, valid_df['ImageId'], masks)


weight_path = 'weights/{}.weights.h5'.format('seg_model')
checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1, mode='max', save_weights_only=True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.2, patience=3, verbose=1, mode='max', min_delta=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef", mode="max", patience=15)

callbacks_list = [checkpoint, early, reduceLROnPlat]


built_model.compile(optimizer=Adam(1e-3), loss=FocalLoss, metrics=[dice_coef, 'binary_accuracy'])

history = built_model.fit(
    train_dfrm,
    validation_data=valid_dfrm,
    callbacks=callbacks_list,
    verbose=1,
    epochs=min(10, MAX_EPOCHS),
    steps_per_epoch=10,
    validation_steps=10
)

built_model.load_weights(weight_path)
built_model.save(MODEL_PATH)