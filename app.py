from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# 全域變數：遊戲記錄與模型狀態
game_results = []  # 存放歷史數據 (例如：['B', 'P', ...])
model = RandomForestClassifier()  # 使用隨機森林模型
is_model_ready = False  # 模型是否訓練完成

# ------------------------------------------
# 功能函數
# ------------------------------------------
def prepare_training_data(results, window_size=3):
    """利用滑動窗口方法準備訓練數據"""
    if len(results) <= window_size:
        return [], []
    X, y = [], []
    for i in range(window_size, len(results)):
        X.append(results[i - window_size:i])
        y.append(results[i])
    return X, y

def encode_data(data):
    """將 'B' 與 'P' 轉換為數字 (B: 1, P: 0)"""
    return [1 if x == "B" else 0 for x in data]

def decode_data(data):
    """將數字轉換回文字"""
    return "B" if data == 1 else "P"

def train_model():
    """訓練隨機森林模型"""
    global is_model_ready
    if len(game_results) < 5:
        return False
    X, y = prepare_training_data(game_results)
    if not X or not y:
        return False
    X = np.array([encode_data(x) for x in X])
    y = np.array(encode_data(y))
    model.fit(X, y)
    is_model_ready = True
    return True

def predict_next():
    """模型預測下一次結果"""
    if not is_model_ready or len(game_results) < 3:
        return "未知"
    last_sequence = game_results[-3:]
    X = np.array([encode_data(last_sequence)])
    prediction = model.predict(X)[0]
    return decode_data(prediction)

# ------------------------------------------
# Flask 路由
# ------------------------------------------
@app.route('/')
def index():
    # 透過 index.html 模板顯示前端介面
    return render_template('index.html')

@app.route('/update_result', methods=['POST'])
def update_result_endpoint():
    global game_results
    data = request.get_json()
    result = data.get('result')
    if result not in ['B', 'P']:
        return jsonify({'error': '無效的輸入'}), 400

    game_results.append(result)

    # 準備回傳的歷史記錄 (直接傳整個列表，前端可進行處理)
    history = game_results[:]

    # 當記錄不足時顯示提示訊息
    if len(game_results) < 3:
        prediction_text = "數據不足，無法有效預測"
    else:
        if train_model():
            prediction = predict_next()
            if prediction != "未知":
                prediction_text = "下一次可能是：" + ("莊" if prediction == "B" else "閒")
            else:
                prediction_text = "數據不足，無法預測"
        else:
            prediction_text = "數據不足，無法有效預測"

    return jsonify({'history': history, 'prediction': prediction_text})

@app.route('/reset', methods=['POST'])
def reset_game():
    global game_results, is_model_ready
    game_results = []
    is_model_ready = False
    return jsonify({
        'history': "請輸入更多數據以便系統分析",
        'prediction': "下一次可能是：請輸入更多數據以便系統分析"
    })

if __name__ == '__main__':
    app.run(debug=True)
