# PM Internship Scheme Web Application

A sophisticated Flask-based web application for the PM Internship Scheme featuring AI-powered internship recommendations, multi-language support, and modern responsive design. The application provides personalized internship matching using machine learning algorithms and offers a comprehensive user experience for internship discovery and management.

## 🚀 Key Features

### Core Functionality
- **AI-Powered Recommendations**: Advanced machine learning system using TF-IDF vectorization and cosine similarity
- **Smart Search & Filtering**: Enhanced search with skills, location, and status-based filtering
- **User Authentication**: Secure registration/login with password hashing and session management
- **Profile Management**: Comprehensive user profiles with education and location data
- **Internship Management**: Save/star internships with similarity scoring and tracking

### Advanced Features
- **Multi-Language Support**: English and Hindi interface with dynamic language switching
- **Responsive Design**: Mobile-first Bootstrap 5 interface with custom styling
- **Real-time Filtering**: Filter internships by status (ongoing, future, expired)
- **Enhanced Sorting**: Sort by date, deadline, stipend, or ML similarity scores
- **Modern UI/UX**: Gradient backgrounds, smooth animations, and intuitive navigation

### Machine Learning System
- **Content-Based Filtering**: Analyzes job titles, skills, company data, and location preferences
- **Multi-Feature Processing**: Combines text, categorical, and numeric features
- **Dynamic Recommendations**: Real-time personalized suggestions based on user profile
- **Model Persistence**: Trained model storage and retrieval using joblib
- **Enhanced Matching**: Prioritizes job title and location matches for better relevance

## 🛠️ Technology Stack

### Backend
- **Flask**: Web framework with modular architecture
- **SQLAlchemy**: ORM for database operations with Flask-SQLAlchemy
- **Flask-Login**: User session management and authentication
- **Flask-WTF**: Form handling and CSRF protection
- **Flask-Migrate**: Database migration support

### Machine Learning
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **pandas**: Data processing and manipulation
- **numpy**: Numerical computations
- **scipy**: Sparse matrix operations
- **joblib**: Model serialization and persistence

### Frontend
- **Bootstrap 5**: Responsive CSS framework
- **Font Awesome**: Icon library
- **Google Fonts (Inter)**: Modern typography
- **Custom CSS**: Advanced styling with CSS variables and animations

### Database
- **SQLite**: Lightweight database (easily migrated to PostgreSQL/MySQL)
- **User Model**: Authentication and profile data
- **StarredInternship Model**: ML recommendations and user preferences

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd job_search
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables (optional but recommended):**
   ```bash
   # Windows
   set SECRET_KEY=your-secure-secret-key-here
   
   # macOS/Linux
   export SECRET_KEY=your-secure-secret-key-here
   ```

### 🤖 Machine Learning Model Setup

1. **Train the ML recommendation model:**
   ```bash
   cd ml_model
   python internship_reccomander.py
   ```
   This will create `internship_recommender_model.pkl` for AI-powered recommendations.

2. **Verify model training:**
   The script will automatically test the trained model with sample user profiles.

## 🚀 Running the Application

1. **Initialize the database:**
   ```bash
   python app.py
   ```
   The database will be automatically created on first run.

2. **Access the application:**
   Open your web browser and navigate to `http://127.0.0.1:5000`

3. **Create an account:**
   - Click "Sign Up" to create a new user account
   - Fill in your profile information for better ML recommendations
   - Start searching for internships with AI-powered suggestions!

## 📁 Project Structure

```
job_search/
├── app.py                          # Main Flask application
├── models.py                       # Database models (User, StarredInternship)
├── forms.py                        # WTForms for user input validation
├── requirements.txt                # Python dependencies
├── translations.json               # Multi-language support (EN/HI)
├── README.md                       # Project documentation
│
├── ml_model/                       # Machine Learning System
│   ├── internship_reccomander.py   # ML recommendation engine
│   ├── Internship.csv              # Training dataset
│   ├── internship_recommender_model.pkl  # Trained model
│   └── __pycache__/                # Python cache files
│
├── static/                         # Static assets
│   └── css/
│       └── main.css                # Custom styling with CSS variables
│
├── templates/                      # Jinja2 HTML templates
│   ├── layout.html                 # Base template with navigation
│   ├── home.html                   # Main search and recommendations page
│   ├── login.html                  # User authentication
│   ├── signup.html                 # User registration
│   ├── dashboard.html              # User dashboard with saved internships
│   ├── profile.html                # User profile display
│   ├── edit_profile.html           # Profile editing form
│   └── internship_card.html        # Reusable internship display component
│
├── instance/                       # Instance-specific files
│   └── internship_portal.db        # SQLite database
│
└── venv/                          # Virtual environment (created during setup)
```

## 🔧 Configuration

### Environment Variables
- `SECRET_KEY`: Flask secret key for session security (auto-generated if not set)
- `SQLALCHEMY_DATABASE_URI`: Database connection string (defaults to SQLite)

### Database Models
- **User**: Stores authentication and profile data
- **StarredInternship**: Stores user's saved ML recommendations with similarity scores

## 🌟 Usage Guide

### For Users
1. **Register**: Create an account with your educational and location details
2. **Search**: Use the AI-powered search to find relevant internships
3. **Filter**: Apply filters for ongoing, future, or expired internships
4. **Save**: Star interesting internships for later review
5. **Manage**: View saved internships in your dashboard

### For Developers
1. **ML Model**: Retrain the model with new data by running the ML script
2. **Database**: Use Flask-Migrate for schema changes
3. **Translations**: Add new languages by updating `translations.json`
4. **Styling**: Modify CSS variables in `main.css` for theme customization

## 🚀 Advanced Features

### Machine Learning Pipeline
- **Data Processing**: Automatic text cleaning and feature extraction
- **Model Training**: TF-IDF + Cosine Similarity for content-based filtering
- **Real-time Recommendations**: Dynamic user profile matching
- **Enhanced Sorting**: Intelligent ranking based on job title and location matches

### Multi-Language Support
- Dynamic language switching between English and Hindi
- Comprehensive translation system for UI elements
- Maintains search parameters across language changes

### Responsive Design
- Mobile-first Bootstrap 5 implementation
- Custom CSS with modern gradients and animations
- Optimized for all screen sizes and devices

## 🔮 Future Enhancements

- **Advanced ML Features**: Collaborative filtering and hybrid recommendation systems
- **Real-time Notifications**: Email/SMS alerts for new matching internships
- **Application Tracking**: Complete internship application workflow
- **Resume Integration**: PDF resume upload and parsing
- **Company Profiles**: Detailed company information and reviews
- **Analytics Dashboard**: User engagement and recommendation performance metrics
- **API Integration**: Connect with external job boards and company APIs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For support, email support@pminternship.gov.in or create an issue in the repository.