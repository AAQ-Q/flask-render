from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# 初始化遊戲數據和模型
game_results = []
model = RandomForestClassifier()
is_model_ready = False

def prepare_training_data(results, window_size=3):
    """準備訓練數據 (滑動窗口方法)"""
    if len(results) <= window_size:
        return [], []
    X, y = [], []
    for i in range(window_size, len(results)):
        X.append(results[i - window_size:i])
        y.append(results[i])
    return X, y

def encode_data(data):
    """將 BP 轉換為數字 (B: 1, P: 0)"""
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

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/update_result', methods=['POST'])
def update_result():
    global game_results
    data = request.json
    result = data.get("result")
    
    if result:
        game_results.append(result)
    
    if len(game_results) < 3:
        return jsonify({"prediction": "數據不足，無法有效預測", "history": game_results})
    
    was_trained = train_model()
    if not was_trained:
        return jsonify({"prediction": "數據不足，無法有效預測", "history": game_results})

    next_prediction = predict_next()
    return jsonify({
        "prediction": f"下一次可能是：{'莊' if next_prediction == 'B' else '閒'}",
        "history": game_results
    })

@app.route('/reset', methods=['POST'])
def reset():
    global game_results, is_model_ready
    game_results = []
    is_model_ready = False
    return jsonify({"message": "遊戲記錄已重置！", "history": game_results})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
