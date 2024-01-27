import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


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
    image_path = 'inputs/test/musk.jpg'
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


if __name__ == '__main__':
    testHayao_36()

