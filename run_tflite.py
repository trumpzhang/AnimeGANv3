import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

print(tf.__version__)
tf.compat.v1.enable_eager_execution()



def testHayao_36():
    # 1. 加载 TensorFlow Lite 模型
    model_path = 'models/AnimeGANv3_Hayao_36.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 2. 获取输入和输出张量的索引
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)

    # 3. 加载图片并进行预处理
    image_path = 'inputs/test/v3_17.jpg'
    image = Image.open(image_path)
    image = image.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    image = np.expand_dims(image, axis=0)
    image = (image.astype(np.float32) - 127.5) / 127.5  # 这是一个简单的归一化步骤，确保输入符合模型的期望范围

    # 4. 设置输入张量的值
    interpreter.set_tensor(input_details[0]['index'], image)

    # 5. 运行推理
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    # 6. 获取输出结果
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # 现在，output_data 包含了模型对输入图片的预测结果
    # print("Model Output:", output_data)

    # 7. 打印推理时间
    inference_time = end_time - start_time
    print("Inference Time: {:.2f} seconds".format(inference_time))

    # 8. 可视化处理后的图片
    plt.imshow(np.squeeze(output_data))
    plt.title('Processed Image')
    plt.show()


# [Style Transfer]
# Function to load an image from a file, and add a batch dimension.
style_predict_path="models/style_transfer/predict_float16.tflite"
style_path = "inputs/test/style_trans/style.png"
content_path = "inputs/test/style_trans/content.png"

def load_img(path_to_img):
  img = tf.io.read_file(path_to_img)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = img[tf.newaxis, :]

  return img

def load_img_np(image_path, input_details):
    image = Image.open(image_path)
    image = image.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    image = (np.array(image).astype(np.float32) - 127.5) / 127.5
    image = np.expand_dims(image, axis=0)
    return image


# Function to pre-process by resizing an central cropping it.
def preprocess_image(image, target_dim):
  # Resize the image so that the shorter dimension becomes 256px.
  shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
  short_dim = min(shape)
  scale = target_dim / short_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  image = tf.image.resize(image, new_shape)

  # Central crop the image.
  image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

  return image


# Load the input images.


# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=style_predict_path)

  # Set model input.
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

  # Calculate style bottleneck.
  start_time = time.time()
  interpreter.invoke()
  end_time = time.time()
  print("Inference Time: {:.2f} seconds".format(end_time - start_time))


  style_bottleneck = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return style_bottleneck


def process_style_img():
    style_image = load_img(style_path)
    preprocessed_style_image = preprocess_image(style_image, 256)
    style_bottleneck = run_style_predict(preprocessed_style_image)
    print('Style Bottleneck Shape:', style_bottleneck.shape)




if __name__ == '__main__':
    testHayao_36()
    # process_style_img()

