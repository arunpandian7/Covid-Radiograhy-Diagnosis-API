import tensorflow as tf
import tensorflow_hub as tfhub


MODEL_ARCH_PATH = "models/configs/tf_hub_effv2/"
MODEL_BASE_PATH = "models/weights/efficient-classifer"

LABELS = ['Negative', 'Typical Appearance', 'Indeterminate', 'Atypical']

devices = tf.config.list_physical_devices()
tf.config.experimental.set_memory_growth(
    devices[1], True
)

on_gpu = False
ensemble = True

# Custom wrapper class to load the right pretrained weights explicitly from the local directory
class KerasLayerWrapper(tfhub.KerasLayer):
    def __init__(self, handle, **kwargs):
        handle = tfhub.KerasLayer(tfhub.load(MODEL_ARCH_PATH))
        super().__init__(handle, **kwargs)

device = '/device:CPU:0' if not on_gpu else '/device:GPU:0'

with tf.device(device):
    models = []
    models0 = tf.keras.models.load_model(f'{MODEL_BASE_PATH}/model0.h5',
                                custom_objects={'KerasLayer': KerasLayerWrapper})
    models.append(models0)

    if ensemble:
        models1 = tf.keras.models.load_model(f'{MODEL_BASE_PATH}/model1.h5',
                                                custom_objects={'KerasLayer': KerasLayerWrapper})
        models2 = tf.keras.models.load_model(f'{MODEL_BASE_PATH}/model2.h5',
                                                custom_objects={'KerasLayer': KerasLayerWrapper})
        models3 = tf.keras.models.load_model(f'{MODEL_BASE_PATH}/model3.h5',
                                                custom_objects={'KerasLayer': KerasLayerWrapper})
        models4 = tf.keras.models.load_model(f'{MODEL_BASE_PATH}/model4.h5',
                                                custom_objects={'KerasLayer': KerasLayerWrapper})
        models.append(models1)
        models.append(models2)
        models.append(models3)
        models.append(models4)



def read_tf_image(path, target_size=(768, 768)):
    file_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(file_bytes, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, target_size)
    return img


def inference(img):
    with tf.device(device):
        img = read_tf_image(img)
        pred = sum([model.predict(tf.expand_dims(img, axis=0)) for model in models]) / len(models)
        pred_label = LABELS[pred.argmax()]
        confidence = pred[0][pred.argmax()]
    return pred_label, confidence
