from keras_applications import get_submodules_from_kwargs
import tensorflow as tf
from utils import *
import math
from classification_models.tfkeras import Classifiers




# ---------------------------------------------------------------------
#  Utility 
# ---------------------------------------------------------------------

tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math
backend = None
layers = None
models = None
keras_utils = None

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1


    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x
    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):

        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x
    return layer


# ---------------------------------------------------------------------
#  TransOnet Decoder
# ---------------------------------------------------------------------

def build_ton(
        backbone,
        decoder_block,
        skip_connection_layers,
        transormer_num,
        pos_embedding,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
):
    
    global backend, layers, models, keras_utils
    from tensorflow.keras import backend, layers, models
    
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])



#---------------------------------------
# Bridge: Transformer unit
#-------------------------------------

    if pos_embedding:
        x = PatchEmbedding()(x)
    else:
        x = tfkl.Reshape((x.shape[1] * x.shape[2], 512))(x)
        x = tfkl.Dense(512, input_shape=(512,), activation=None)(x)



    for i in range(transormer_num):
        x, _ = TransformerBlock(
            n_heads = 16, #number of heads
            mlp_dim = 3072,
            dropout= 0.1,
            name=f"Transformer/encoderblock_{i}"
            )(x)

    x = tfkl.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(x)

    n_patch_sqrt = int(math.sqrt(x.shape[1]))

    x = tfkl.Reshape(
        target_shape=[n_patch_sqrt, n_patch_sqrt, 512])(x)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Conv2D(512, 1, activation='relu')(x)


    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = models.Model(input_, x)

    return model


# ---------------------------------------------------------------------
#  TransOnet Model
# ---------------------------------------------------------------------

def TransOnet(
        backbone_name='vgg16',
        input_shape=(512, 512, 3),
        classes=1,
        activation='sigmoid',
        weights=None,
        encoder_weights='imagenet',
        transormer_num = 1,
        pos_embedding = False,
        encoder_freeze=False,
        encoder_features='default',
        decoder_block=DecoderUpsamplingX2Block,
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        **kwargs
):
   

    backbone_net, _ = Classifiers.get('resnet34')
    backbone = backbone_net(input_shape=input_shape, weights=encoder_weights, include_top=False)


    skip_connection_layers = (
        'stage4_unit1_relu1', 
        'stage3_unit1_relu1', 
        'stage2_unit1_relu1', 
        'relu0'
        )


    if encoder_features == 'default':
        encoder_features = skip_connection_layers # backbone.get_layer(layer_name).output for layer_name in skip_connection_layers
    

    model = build_ton(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        pos_embedding = pos_embedding,
        transormer_num= transormer_num,
        classes=classes,
        activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )



    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model
