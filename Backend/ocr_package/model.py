from datetime import datetime
from ocr_package import db, login_manager
from flask_login import UserMixin
from ocr_package.forms import ValidationError

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False)
    surname = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20),  nullable=False)
    password = db.Column(db.String(60), nullable=False)
    created_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    cards = db.relationship('Card', backref='user', lazy=True)
   
    def __repr__(self):
        return f"User('{self.name}', '{self.surname}', '{self.email}', '{self.phone}')"

class Card(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    card_number = db.Column(db.String(30), nullable=False)
    card_holder = db.Column(db.String(20), nullable=False)
    valid_date = db.Column(db.String(10), nullable=False)
    image_file = db.Column(db.String(20), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Card('{self.card_number}', '{self.card_holder}', '{self.valid_date}')"
    


