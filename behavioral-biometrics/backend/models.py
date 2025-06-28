import os
import sys
from flask import Flask, request, jsonify
from django import setup
from app.models import SessionLog
from datetime import datetime
import requests

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
setup()

app = Flask(__name__)

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
        is_fraud=data.get('is_fraud', False)
    )
    log.save()

    return jsonify({'fraud': int(log.is_fraud)})

@app.route('/logs')
def get_logs():
    logs = SessionLog.objects.order_by('-id')[:50]
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
