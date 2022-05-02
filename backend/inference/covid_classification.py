import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time

import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np

from backend import config

MODEL_ARCH_PATH = "models/configs/tf_hub_effv2/"
MODEL_BASE_PATH = "models/weights/efficient-classifer"
target_size=(768, 768)

LABELS = ['Negative', 'Typical Appearance', 'Indeterminate', 'Atypical']

devices = tf.config.list_physical_devices()
tf.config.experimental.set_memory_growth(
    devices[1], True
)

# Custom wrapper class to load the right pretrained weights explicitly from the local directory
class KerasLayerWrapper(tfhub.KerasLayer):
    def __init__(self, handle, **kwargs):
        handle = tfhub.KerasLayer(tfhub.load(MODEL_ARCH_PATH))
        super().__init__(handle, **kwargs)

class CovidConditionClassifier:
    def __init__(self, ensemble=False, use_gpu=False) -> None:
        self.device = '/device:CPU:0' if not use_gpu else '/device:GPU:0'
        with tf.device(self.device):
            print("Loading Classification Models......")
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

        print("Successfully loaded Classification Models.....")
        self.models = models

    def inference(self, img: np.ndarray):
        with tf.device(self.device):
            img = tf.convert_to_tensor(img, tf.float32) / 255.0
            img = tf.image.resize(img, target_size)
            start = time.time()
            pred = sum([model.predict(tf.expand_dims(img, axis=0)) for model in self.models]) / len(self.models)
            end = time.time()
            pred_label = LABELS[pred.argmax()]
            confidence = pred[0][pred.argmax()]
            confidence = confidence.astype(float)
            return pred_label, confidence, round(end-start, 10)

condition_classifer = CovidConditionClassifier(ensemble=config.ENSEMBLE, use_gpu=config.USE_GPU)
