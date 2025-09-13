from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

# Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    college = db.Column(db.String(150))
    education = db.Column(db.String(100))
    date_created = db.Column(db.DateTime, default=datetime.utcnow)


class StarredInternship(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # ML recommendation data
    ml_id = db.Column(db.String(50), nullable=False)  # e.g., "ml_123"
    title = db.Column(db.String(255), nullable=False)
    company = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255))
    sector = db.Column(db.String(255))
    duration = db.Column(db.String(100))
    stipend = db.Column(db.String(100))
    description = db.Column(db.Text)
    skills = db.Column(db.Text)
    similarity_score = db.Column(db.Float)
    
    # Dates
    starred_date = db.Column(db.DateTime, default=datetime.utcnow)
    posted_date = db.Column(db.DateTime)
    application_deadline = db.Column(db.DateTime)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('starred_internships', lazy='dynamic'))
