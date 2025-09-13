
import os
from flask import Flask, render_template, redirect, url_for, flash, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from forms import LoginForm, SignupForm, SearchForm, EditProfileForm
import json
# from transformers import pipeline  # Commented out for now
from flask_migrate import Migrate
from models import db
from datetime import datetime

import pandas as pd
from ml_model.internship_reccomander import InternshipRecommender
import os


# Initialize Flask app
app = Flask(__name__)

# Load the trained ML model
model_path = os.path.join('ml_model', 'internship_recommender_model.pkl')
trained_recommender = None
try:
    if os.path.exists(model_path):
        trained_recommender = InternshipRecommender.load_model(model_path)
        print(f"ML model loaded successfully from {model_path}")
    else:
        print(f"ML model not found at {model_path}")
except Exception as e:
    print(f"Error loading ML model: {e}")
    trained_recommender = None
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///internship_portal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Import models first
from models import db, User, StarredInternship
# Initialize SQLAlchemy with app
db.init_app(app)
migrate = Migrate(app, db)   # 🔑 this enables `flask db` commands

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
# Initialize translation model (disabled for now)
translator = None
# Load translations for static text from a JSON file
try:
    with open('translations.json', 'r', encoding='utf-8') as f:
        translations = json.load(f)
except FileNotFoundError:
    print("translations.json not found. Static text will not be translated.")
    translations = {}
@app.before_request
def set_language():
    """Sets the default language to English if not already in session."""
    if 'lang' not in session:
        session['lang'] = 'en'
# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Login unsuccessful. Please check email and password', 'danger')
    
    return render_template('login.html', 
                           title=translations.get(session['lang'], {}).get('login_title', 'Login'), 
                           form=form,
                           static_text=translations.get(session['lang'], {}))
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    form = SignupForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email already registered. Please use a different email or login.', 'danger')
            return render_template('signup.html', 
                                   title=translations.get(session['lang'], {}).get('signup_title', 'Sign Up'), 
                                   form=form,
                                   static_text=translations.get(session['lang'], {}))
        
        hashed_password = generate_password_hash(form.password.data)
        user = User(
            name=form.name.data,
            email=form.email.data,
            password=hashed_password,
            location=form.location.data,
            phone=form.phone.data,
            college=form.college.data,
            education=form.education.data
        )
        
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        flash('Welcome! Your account has been created successfully.', 'success')
        return redirect(url_for('home'))
    
    return render_template('signup.html', 
                           title=translations.get(session['lang'], {}).get('signup_title', 'Sign Up'), 
                           form=form,
                           static_text=translations.get(session['lang'], {}))
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/set_lang/<lang_code>')
def set_lang(lang_code):
    """Route to set the user's language preference and redirect back to the correct page with search params."""
    if lang_code in translations:
        session['lang'] = lang_code

    # Collect current query parameters (query, search_type, etc.)
    query_params = request.args.to_dict()

    # Always redirect to home with whatever params we have
    return redirect(url_for('home', **query_params))



@app.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    current_lang = session.get('lang', 'en')
    static_text = translations.get(current_lang, {})
    search_form = SearchForm()
    
    # --- Default values ---
    query = request.args.get('query', '').strip()
    skills = request.args.get('skills', '').strip()
    city = request.args.get('city', '').strip()
    search_type = request.args.get('search_type')
    filter_list = request.args.getlist('filter_status')  # ✅ multiple checkboxes
    sort_by = request.args.get('sort_by', 'posted_date')  # Default sort by posted date
    internships = []
    ongoing, expired, future = [], [], []
    today = datetime.utcnow().date()

    # --- Handle search (POST → redirect to GET for clean URLs) ---
    if request.method == 'POST' and search_form.validate_on_submit():
        query = search_form.query.data.strip()
        skills = search_form.skills.data.strip() if search_form.skills.data else ''
        city = search_form.city.data.strip() if search_form.city.data else ''
        search_type = search_form.search_type.data
        return redirect(url_for('home', query=query, skills=skills, 
                               city=city, search_type=search_type, sort_by=sort_by))

    # Pre-populate form
    search_form.query.data = query
    search_form.skills.data = skills
    search_form.city.data = city
    search_form.search_type.data = search_type

    # --- Perform search only if query is provided and ML model is available ---
    if query and trained_recommender:
        # Use ML model for recommendations
        try:
            # Create comprehensive user profile from all inputs
            combined_skills = f"{query} {skills}".strip()
            combined_location = city.strip() or current_user.location or ''
            
            # Use default values for stipend and duration since they're not input fields
            stipend_val = 10000  # Default minimum stipend
            duration_val = 6     # Default duration in months
            
            user_profile = {
                'Skills': combined_skills or 'General',
                'Preferred Cities': combined_location,
                'Job Title': query or 'Internship',
                'Min Stipend': stipend_val,
                'Duration': duration_val
            }
            
            # Get recommendations from ML model
            recommendations = trained_recommender.get_recommendations_api(user_profile, top_n=20)
            
            # Convert ML recommendations to internship objects with varied dates
            internships = []
            import random
            for i, rec in enumerate(recommendations['recommendations']):
                row_data = rec['row']
                
                # Create varied start/end dates for filtering
                base_date = datetime.utcnow()
                random.seed(rec['index'])  # Consistent randomness based on index
                
                # Assign different statuses: 30% ongoing, 40% future, 30% expired
                status_rand = random.random()
                if status_rand < 0.3:  # Ongoing
                    start_date = base_date - timedelta(days=random.randint(1, 30))
                    end_date = base_date + timedelta(days=random.randint(30, 120))
                elif status_rand < 0.7:  # Future
                    start_date = base_date + timedelta(days=random.randint(1, 60))
                    end_date = start_date + timedelta(days=random.randint(60, 180))
                else:  # Expired
                    end_date = base_date - timedelta(days=random.randint(1, 30))
                    start_date = end_date - timedelta(days=random.randint(60, 120))
                
                # Extract numeric stipend for sorting
                stipend_numeric = row_data.get('Stipend', 10000)
                
                internship_data = {
                    'id': f"ml_{rec['index']}",  # Unique ID for ML recommendations
                    'title': row_data.get('Job Title', 'ML Recommended Position'),
                    'company': row_data.get('Company', 'ML Company'),
                    'location': row_data.get('Cities', 'Various'),
                    'sector': 'Technology',  # Default sector
                    'duration': f"{row_data.get('Duration', 6)} months",
                    'stipend': f"₹{stipend_numeric}/month",
                    'stipend_numeric': stipend_numeric,  # For sorting
                    'description': f"Skills: {row_data.get('Skills', 'Various skills')}. Similarity Score: {rec['similarity']:.3f}",
                    'posted_date': start_date - timedelta(days=random.randint(1, 14)),
                    'start_date': start_date,
                    'end_date': end_date,
                    'application_deadline': end_date - timedelta(days=random.randint(1, 14)),
                    'ml_similarity': rec['similarity'],
                    'is_ml_recommendation': True  # Flag to identify ML recommendations
                }
                
                # Create a simple object to mimic Internship model
                class MLInternship:
                    def __init__(self, data):
                        for key, value in data.items():
                            setattr(self, key, value)
                
                internships.append(MLInternship(internship_data))
            
            # Store search results in session for star functionality
            session['current_search_results'] = {
                'internships': [
                    {
                        'id': internship.id,
                        'title': internship.title,
                        'company': internship.company,
                        'location': internship.location,
                        'sector': internship.sector,
                        'duration': internship.duration,
                        'stipend': internship.stipend,
                        'description': internship.description,
                        'ml_similarity': internship.ml_similarity
                    } for internship in internships
                ]
            }
            
            # --- Categorize ML results ---
            for i in internships:
                if i.start_date and i.end_date:
                    if i.start_date.date() <= today <= i.end_date.date():
                        ongoing.append(i)     # ✅ ongoing
                    elif today < i.start_date.date():
                        future.append(i)      # ✅ not started yet
                    elif today > i.end_date.date():
                        expired.append(i)     # ✅ already ended
                else:
                    ongoing.append(i)  # fallback if dates missing
                
        except Exception as e:
            print(f"ML recommendation error: {e}")
            flash('ML recommendations temporarily unavailable. Please try again.', 'warning')
            internships = []
    elif query and not trained_recommender:
        flash('ML model not available. Please check the system configuration.', 'error')
        internships = []

    # Default (before filters)
    internships = ongoing + future + expired

    # --- Apply filters ---
    if filter_list:
        filtered = []
        if "ongoing" in filter_list:
            filtered.extend(ongoing)
        if "future" in filter_list:
            filtered.extend(future)
        if "expired" in filter_list:
            filtered.extend(expired)
        internships = filtered

    # --- Apply enhanced sorting for ML recommendations ---
    if internships and any(hasattr(i, 'is_ml_recommendation') for i in internships):
        # Enhanced sorting for ML recommendations: prioritize job title + city matches
        def enhanced_ml_sort_key(internship):
            if not hasattr(internship, 'is_ml_recommendation'):
                return (0, 0, getattr(internship, 'ml_similarity', 0))
            
            # Check job title match (case-insensitive)
            title_match = 0
            if query and hasattr(internship, 'title'):
                if query.lower() in internship.title.lower():
                    title_match = 2  # Exact job title match gets highest priority
                elif any(word.lower() in internship.title.lower() for word in query.split()):
                    title_match = 1  # Partial job title match gets medium priority
            
            # Check city match (case-insensitive)
            city_match = 0
            if city and hasattr(internship, 'location'):
                if city.lower() in internship.location.lower():
                    city_match = 1  # City match gets priority
            
            # Return tuple for sorting: (title_match, city_match, similarity_score)
            # Higher values = better matches = sorted first
            similarity = getattr(internship, 'ml_similarity', 0)
            return (title_match, city_match, similarity)
        
        # Sort ML recommendations with enhanced logic
        internships.sort(key=enhanced_ml_sort_key, reverse=True)
    
    # --- Apply regular sorting ---
    elif internships:  # Only sort if we have internships
        if sort_by == 'posted_date':
            internships.sort(key=lambda x: getattr(x, 'posted_date', datetime.min), reverse=True)
        elif sort_by == 'deadline':
            internships.sort(key=lambda x: getattr(x, 'application_deadline', datetime.max), reverse=False)
        elif sort_by == 'stipend':
            # Sort by stipend (handle ML internships)
            def get_stipend_value(internship):
                # For ML recommendations, use the numeric value directly
                if hasattr(internship, 'stipend_numeric'):
                    return internship.stipend_numeric
                return 0
            internships.sort(key=get_stipend_value, reverse=True)
        elif sort_by == 'similarity' and any(hasattr(i, 'ml_similarity') for i in internships):
            # Sort by ML similarity score
            internships.sort(key=lambda x: getattr(x, 'ml_similarity', 0), reverse=True)

    # --- No query → just search bar (don’t show internships) ---
    else:
        internships = []

    # --- Handle translations (unchanged from your code) ---
    if current_lang == 'hi' and translator:
        updated = False
        for internship in internships:
            try:
                if not getattr(internship, 'title_hi', None) and internship.title:
                    internship.title_hi = translator(internship.title)[0]['translation_text']
                    updated = True
                if not getattr(internship, 'description_hi', None) and internship.description:
                    desc_to_translate = (
                        internship.description[:200] + "..."
                        if len(internship.description) > 200 else internship.description
                    )
                    internship.description_hi = translator(desc_to_translate)[0]['translation_text']
                    updated = True
            except Exception as e:
                print(f"Translation error: {e}")
                continue
        if updated:
            try:
                db.session.commit()
            except Exception as e:
                print(f"Database commit error: {e}")
                db.session.rollback()

    # Get user's starred internships for UI state
    starred_internships = current_user.starred_internships.all()
    user_saved_ids = [starred.ml_id for starred in starred_internships]
    
    return render_template(
        'home.html',
        search_form=search_form,
        query=query,
        internships=internships,
        ongoing_internships=ongoing,
        future_internships=future,
        expired_internships=expired,
        static_text=static_text,
        current_lang=current_lang,
        active_filters=filter_list,
        sort_by=sort_by,
        user_saved_ids=user_saved_ids,
        user_applied_ids=[]  # No longer needed
    )



@app.route('/profile')
@login_required
def profile():
    current_lang = session.get('lang', 'en')
    static_text = translations.get(current_lang, {})
    
    # Get user's starred internships
    starred_internships = current_user.starred_internships.all()
    
    # Mock user statistics
    user_stats = {
        'total_searches': 0,
        'profile_views': 1,
        'member_since': 'Recently',
        'saved_internships_count': len(starred_internships)
    }
    
    return render_template('profile.html', 
                         title=static_text.get('profile_title', 'Profile'),
                         static_text=static_text,
                         user_stats=user_stats,
                         saved_internships=starred_internships)

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    current_lang = session.get('lang', 'en')
    static_text = translations.get(current_lang, {})
    form = EditProfileForm()
    
    if form.validate_on_submit():
        current_user.name = form.name.data
        current_user.email = form.email.data
        current_user.phone = form.phone.data
        current_user.location = form.location.data
        current_user.college = form.college.data
        current_user.education = form.education.data
        
        try:
            db.session.commit()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            db.session.rollback()
            flash('Error updating profile. Please try again.', 'error')
    
    # Pre-populate form with current user data
    form.name.data = current_user.name
    form.email.data = current_user.email
    form.phone.data = current_user.phone
    form.location.data = current_user.location
    form.college.data = current_user.college
    form.education.data = current_user.education
    
    return render_template('edit_profile.html',
                         title=static_text.get('edit_profile', 'Edit Profile'),
                         form=form,
                         static_text=static_text)


@app.route('/save_internship/<internship_id>', methods=['POST'])
@login_required
def save_internship(internship_id):
    # Handle ML recommendations (string IDs starting with 'ml_')
    if str(internship_id).startswith('ml_'):
        # Check if already starred
        existing = StarredInternship.query.filter_by(
            user_id=current_user.id, 
            ml_id=internship_id
        ).first()
        
        if existing:
            # Remove from starred
            db.session.delete(existing)
            action = 'removed'
            flash('Internship removed from saved list!', 'info')
        else:
            # Get internship data from session if available, otherwise use defaults
            internship_data = session.get('current_search_results', {})
            found_internship = None
            
            # Try to find the specific internship from current search results
            for internship in internship_data.get('internships', []):
                if internship.get('id') == internship_id:
                    found_internship = internship
                    break
            
            if found_internship:
                starred = StarredInternship(
                    user_id=current_user.id,
                    ml_id=internship_id,
                    title=found_internship.get('title', 'AI Recommended Position'),
                    company=found_internship.get('company', 'ML Company'),
                    location=found_internship.get('location', 'Various'),
                    sector=found_internship.get('sector', 'Technology'),
                    duration=found_internship.get('duration', '6 months'),
                    stipend=found_internship.get('stipend', '₹10000/month'),
                    description=found_internship.get('description', 'AI recommended internship'),
                    skills=found_internship.get('skills', 'Various skills'),
                    similarity_score=found_internship.get('ml_similarity', 0.0),
                    posted_date=datetime.utcnow(),
                    application_deadline=datetime.utcnow() + timedelta(days=30)
                )
            else:
                # Fallback with minimal data
                starred = StarredInternship(
                    user_id=current_user.id,
                    ml_id=internship_id,
                    title="AI Recommended Position",
                    company="ML Company",
                    location="Various",
                    sector="Technology",
                    duration="6 months",
                    stipend="₹10000/month",
                    description="AI recommended internship",
                    skills="Various skills",
                    similarity_score=0.0,
                    posted_date=datetime.utcnow(),
                    application_deadline=datetime.utcnow() + timedelta(days=30)
                )
            
            db.session.add(starred)
            action = 'saved'
            flash('Internship saved successfully!', 'success')
        
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'error')
            print(f"Database error: {e}")
        
        return redirect(request.referrer or url_for('home'))
    
    # Invalid internship ID format
    flash('Invalid internship ID', 'error')
    return redirect(request.referrer or url_for('home'))


@app.route('/dashboard')
@login_required
def dashboard():
    current_lang = session.get('lang', 'en')
    static_text = translations.get(current_lang, {})
    
    # Get starred internships from database
    starred_internships = current_user.starred_internships.all()
    
    return render_template('dashboard.html',
                         title=static_text.get('dashboard_title', 'Dashboard'),
                         saved_internships=starred_internships,
                         applied_internships=[],  # No longer needed
                         completed_internships=[],  # No longer needed
                         static_text=static_text)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)