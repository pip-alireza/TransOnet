import glob
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

os.environ['SM_FRAMEWORK'] = 'tf.keras'
print("*****************Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import segmentation_models as sm
from model import TransOnet



# from segmentation_models.models import unet 


BACKBONE = 'resnet34'
# preprocess_input = sm.get_preprocessing(BACKBONE)

path = "test data"
# careful. it may give incompatible shape error using different shapes

X_pixel = 512
Y_pixel = 512
train_img = []
train_msk = []


for im in sorted(glob.glob(path + '/*img.png')):
    image = cv2.imread(im, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (X_pixel, Y_pixel))
    image = image.astype(np.uint8)
    train_img.append(image)

train_images = np.array(train_img)


#train_images=np.expand_dims(train_images, axis=3)
for msk in sorted(glob.glob(path +  '/*mask.png')):
    mask = cv2.imread(msk, cv2.IMREAD_COLOR)
    mask = cv2.resize(mask, (X_pixel, Y_pixel))
    # mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask[mask > 0] = 255 # this turns the image to B&W
    mask = mask/255 # this is necessary for iou since we want the value be btw 0 and 1
    #mask = mask.astype(np.uint8)
    mask = mask[:, :, 0] # for mask it is 3 dimension image (512,512,3) but all 3 are same
    train_msk.append(mask)

train_masks = np.array(train_msk)
train_masks=np.expand_dims(train_masks,axis=3)
x_train, x_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.4)

# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)


# The number of transformer units can be adjust here
model = TransOnet(BACKBONE,input_shape=(512, 512, 3), encoder_weights='imagenet', classes=1, transormer_num = 4, pos_embedding=True, activation='sigmoid')
model.summary()



model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

log_name = "ton_JRT"
hist_log = tf.keras.callbacks.CSVLogger(log_name, separator= ",", append=True)

model.fit(
    x=x_train,
    y=y_train,
    batch_size=5,
    epochs=10,
    shuffle=True,
    validation_data=(x_val, y_val),
    callbacks= [hist_log],
)

# accuracy = model.evaluate(x_val, y_val)

model.save('ton.h5')


