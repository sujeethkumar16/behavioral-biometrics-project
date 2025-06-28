from flask import Flask, request, jsonify
from models import db, SessionLog
import torch
import numpy as np
import requests
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sessions.db'
db.init_app(app)

with app.app_context():
    db.create_all()

# Load the PyTorch model
model = torch.load("pytorch_model.pth")
model.eval()  # Set the model to evaluation mode

def get_geo_info(ip):
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}").json()
        return {'ip': ip, 'city': r.get('city'), 'country': r.get('country')}
    except Exception as e:
        print(f"Error fetching geo info: {e}")
        return {'ip': ip, 'city': 'unknown', 'country': 'unknown'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X = np.array([
        data['avg_speed'],
        data['avg_click_delay'],
        data['movement_variance'],
        data['avg_key_delay']
    ], dtype=np.float32).reshape(1, 1, -1)

    # Convert to PyTorch tensor
    X_tensor = torch.from_numpy(X)

    with torch.no_grad():  # Disable gradient calculation
        prediction = model(X_tensor).item() > 0.5  # Get the prediction

    ip = request.remote_addr
    geo = get_geo_info(ip)

    log = SessionLog(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ip=ip,
        city=geo['city'],
        country=geo['country'],
        avg_speed=data['avg_speed'],
        avg_click_delay=data['avg_click_delay'],
        movement_variance=data['movement_variance'],
        avg_key_delay=data['avg_key_delay'],
        is_fraud=bool(prediction)
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({'fraud': int(prediction)})

@app.route('/logs')
def get_logs():
    logs = SessionLog.query.order_by(SessionLog.id.desc()).limit(50).all()
    return jsonify([
        {
            'time': log.timestamp,
            'ip': log.ip,
            'city': log.city,
            'country': log.country,
            'fraud': log.is_fraud
        } for log in logs
    ])

if __name__ == '__main__':
    app.run(debug=True)
