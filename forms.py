from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class SignupForm(FlaskForm):
    name = StringField('Full Name', validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', 
                                   validators=[DataRequired(), EqualTo('password')])
    location = StringField('Location', validators=[DataRequired()])
    phone = StringField('Phone Number', validators=[DataRequired()])
    college = StringField('College/University', validators=[DataRequired()])
    education = StringField('Field of Study', validators=[DataRequired()])
    submit = SubmitField('Sign Up')

class SearchForm(FlaskForm):
    query = StringField('Search', validators=[DataRequired()])
    skills = StringField('Skills (e.g., Python, Machine Learning)')
    location = StringField('Preferred Location')
    city = StringField('City')
    state = StringField('State')
    min_stipend = StringField('Minimum Stipend (₹)')
    duration = StringField('Duration (months)')
    search_type = SelectField('Search By', 
                            choices=[('ml_recommendation', 'AI Recommendations')],
                            validators=[DataRequired()],
                            default='ml_recommendation')
    submit = SubmitField('Search')

class EditProfileForm(FlaskForm):
    name = StringField('Full Name', validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone Number')
    location = StringField('Location')
    college = StringField('College/University')
    education = StringField('Field of Study')
    submit = SubmitField('Update Profile')