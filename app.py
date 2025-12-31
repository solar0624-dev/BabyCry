import os
import requests
import numpy as np
from flask import Flask, request, jsonify
import tempfile

# 引入深度学习库 (MVP阶段如果没有真实模型文件，这段代码会自动降级为模拟模式)
try:
    import tensorflow as tf
    import librosa
    # 尝试加载同一目录下的模型文件
    if os.path.exists('model.h5'):
        model = tf.keras.models.load_model('model.h5')
        print("✅ 真实 AI 模型加载成功")
        MODE = 'REAL'
    else:
        print("⚠️ 未找到 model.h5，进入【模拟模式】")
        MODE = 'MOCK'
except ImportError:
    print("⚠️ 缺少依赖库，进入【模拟模式】")
    MODE = 'MOCK'

app = Flask(__name__)

# 定义哭声标签 (需要与你模型训练时的标签顺序一致)
LABELS = ['Hunger', 'Pain', 'Sleepy', 'Discomfort', 'Gas']
CN_LABELS = {
    'Hunger': '饥饿', 'Pain': '疼痛', 'Sleepy': '困倦', 
    'Discomfort': '不适', 'Gas': '胀气'
}

def preprocess_audio(file_path):
    """
    真实处理逻辑：读取音频 -> 提取 MFCC 特征
    """
    # 这里是一个标准的特征提取流程示例
    y, sr = librosa.load(file_path, duration=5.0, sr=16000)
    # 提取 MFCC 特征
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    # 调整维度以适配模型输入 (例如: 1, 40)
    return np.expand_dims(mfccs, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        file_url = data.get('url')
        
        if not file_url:
            return jsonify({'code': 400, 'msg': '缺少 url 参数'})

        print(f"正在处理音频: {file_url}")

        # MVP 模拟模式 (当你还没有训练好的 model.h5 时使用)
        if MODE == 'MOCK':
            import time, random
            time.sleep(1) # 模拟计算耗时
            pred_idx = random.randint(0, 4)
            label_en = LABELS[pred_idx]
            prob = round(random.uniform(0.7, 0.99), 2)
            return jsonify({
                'code': 0,
                'data': {
                    'type': label_en,
                    'label_cn': CN_LABELS[label_en],
                    'probability': prob,
                    'mode': 'MOCK_Simulated'
                }
            })

        # --- 以下是真实 AI 推理逻辑 ---
        
        # 1. 下载音频文件到临时目录
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_audio:
            r = requests.get(file_url)
            temp_audio.write(r.content)
            temp_audio.flush() # 确保写入完成
            
            # 2. 预处理音频 (提取特征)
            features = preprocess_audio(temp_audio.name)
            
            # 3. 模型预测
            prediction = model.predict(features)
            predicted_index = np.argmax(prediction, axis=1)[0]
            probability = float(np.max(prediction))
            
            label_en = LABELS[predicted_index]
            
            return jsonify({
                'code': 0,
                'data': {
                    'type': label_en,
                    'label_cn': CN_LABELS[label_en],
                    'probability': probability,
                    'mode': 'REAL_Inference'
                }
            })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'code': 500, 'msg': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
