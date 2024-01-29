import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

# print(tf.__version__)
tf.compat.v1.enable_eager_execution()


def testHayao_36():
    # 1. 加载 TensorFlow Lite 模型
    # model_path = 'models/AnimeGANv3_Hayao_36.tflite'
    model_path = 'models/my/AnimeGANv3_Hayao_36.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 2. 获取输入和输出张量的索引
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)

    # 3. 加载图片并进行预处理
    image_path = 'inputs/test/anime/v3_52.jpg'
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
style_transform_path="models/style_transfer/transfer_float16.tflite"

style_path = "inputs/test/style_trans/style4.png"
content_path = "inputs/test/style_trans/man.jpg"

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

style_image = load_img(style_path)
content_image = load_img(content_path)

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
preprocessed_style_image = preprocess_image(style_image, 256)
preprocessed_content_image = preprocess_image(content_image, 384)


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
  plt.show()


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
  print("Predict Inference Time: {:.2f} seconds".format(end_time - start_time))


  style_bottleneck = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return style_bottleneck

# Run style transform on preprocessed style image
def run_style_transform(style_bottleneck, preprocessed_content_image):
  # Load the model.
  interpreter = tf.lite.Interpreter(model_path=style_transform_path)

  # Set model input.
  input_details = interpreter.get_input_details()
  interpreter.allocate_tensors()

  # Set model inputs.
  interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
  interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
  start_time = time.time()
  interpreter.invoke()
  inference_time = time.time() - start_time
  print("Transfer Inference Time: {:.2f} seconds".format(inference_time))


  # Transform content image.
  stylized_image = interpreter.tensor(
      interpreter.get_output_details()[0]["index"]
      )()

  return stylized_image




def process_style_img():
    style_bottleneck = run_style_predict(preprocessed_style_image)
    stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image)
    # print('stylized_image:', stylized_image)
    # Visualize the output.
    imshow(stylized_image, 'Stylized Image')


if __name__ == '__main__':
    testHayao_36()
    # process_style_img()

