
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class SessionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100))
    ip = db.Column(db.String(100))
    city = db.Column(db.String(100))
    country = db.Column(db.String(100))
    avg_speed = db.Column(db.Float)
    avg_click_delay = db.Column(db.Float)
    movement_variance = db.Column(db.Float)
    avg_key_delay = db.Column(db.Float)
    is_fraud = db.Column(db.Boolean)
