from flask import Flask, render_template, redirect, url_for, flash, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, EmailField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from datetime import datetime
import time
import uuid
import os
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///personality_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Load the trained KMeans model (adjust the path if needed)
KMEANS_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'clustering', 'kmeans_model.joblib')
kmeans_model = joblib.load(KMEANS_MODEL_PATH)
# If you used a scaler during training, load it as well:
# SCALER_PATH = os.path.join(os.path.dirname(__file__), 'clustering', 'scaler.joblib')
# scaler = joblib.load(SCALER_PATH)

# User model (from your existing system)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    
    # Relationship with test sessions
    test_sessions = db.relationship('TestSession', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

# Test session model
class TestSession(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Can be null for guest users
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    intro_elapsed = db.Column(db.Float, nullable=True)
    test_elapsed = db.Column(db.Float, nullable=True)
    end_elapsed = db.Column(db.Float, nullable=True)
    screen_width = db.Column(db.Integer, nullable=True)
    screen_height = db.Column(db.Integer, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    country = db.Column(db.String(50), nullable=True)
    consent = db.Column(db.Boolean, default=False)
    cluster_result = db.Column(db.Integer, nullable=True)  # Store cluster assignment
    
    # Relationship with responses
    responses = db.relationship('Response', backref='session', lazy=True)

class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), db.ForeignKey('test_session.id'), nullable=False)
    question_code = db.Column(db.String(10), nullable=False)
    answer = db.Column(db.Integer, nullable=False)
    response_time_ms = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Forms (from your existing system)
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])

# Test data
QUESTIONS = {
    'EXT': [
        'I am the life of the party.',
        'I don\'t talk a lot.',
        'I feel comfortable around people.',
        'I keep in the background.',
        'I start conversations.',
        'I have little to say.',
        'I talk to a lot of different people at parties.',
        'I don\'t like to draw attention to myself.',
        'I don\'t mind being the center of attention.',
        'I am quiet around strangers.'
    ],
    'EST': [
        'I get stressed out easily.',
        'I am relaxed most of the time.',
        'I worry about things.',
        'I seldom feel blue.',
        'I am easily disturbed.',
        'I get upset easily.',
        'I change my mood a lot.',
        'I have frequent mood swings.',
        'I get irritated easily.',
        'I often feel blue.'
    ],
    'AGR': [
        'I feel little concern for others.',
        'I am interested in people.',
        'I insult people.',
        'I sympathize with others\' feelings.',
        'I am not interested in other people\'s problems.',
        'I have a soft heart.',
        'I am not really interested in others.',
        'I take time out for others.',
        'I feel others\' emotions.',
        'I make people feel at ease.'
    ],
    'CSN': [
        'I am always prepared.',
        'I leave my belongings around.',
        'I pay attention to details.',
        'I make a mess of things.',
        'I get chores done right away.',
        'I often forget to put things back in their proper place.',
        'I like order.',
        'I shirk my duties.',
        'I follow a schedule.',
        'I am exacting in my work.'
    ],
    'OPN': [
        'I have a rich vocabulary.',
        'I have difficulty understanding abstract ideas.',
        'I have a vivid imagination.',
        'I am not interested in abstract ideas.',
        'I have excellent ideas.',
        'I do not have a good imagination.',
        'I am quick to understand things.',
        'I use difficult words.',
        'I spend time reflecting on things.',
        'I am full of ideas.'
    ]
}

TRAIT_NAMES = {
    'EXT': 'Extraversion',
    'EST': 'Emotional Stability',
    'AGR': 'Agreeableness',
    'CSN': 'Conscientiousness',
    'OPN': 'Openness'
}

# Cluster descriptions
CLUSTER_DESCRIPTIONS = {
    0: {
        "name": "Reserved Traditionalists",
        "description": "You tend to be practical, cautious, and prefer familiar routines. You're methodical and careful in your approach to tasks and relationships."
    },
    1: {
        "name": "Outgoing Achievers",
        "description": "You're socially confident, goal-oriented, and driven. You enjoy structure and take pride in your accomplishments."
    },
    2: {
        "name": "Creative Independents",
        "description": "You're imaginative, curious, and value your autonomy. You enjoy exploring new ideas and approaches to problems."
    },
    3: {
        "name": "Compassionate Mediators",
        "description": "You're empathetic, cooperative, and concerned with harmony. You're sensitive to others' feelings and prefer collaborative environments."
    },
    4: {
        "name": "Analytical Problem-Solvers",
        "description": "You're logical, detail-oriented, and value precision. You approach problems systematically and make decisions based on careful analysis."
    }
}

QUESTIONS_PER_PAGE = 10  # You can adjust this

# User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes from your existing login system
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page if next_page else url_for('dashboard'))
        else:
            flash('Login failed. Please check your username and password.', 'danger')
    
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        # Check if username or email already exists
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists. Please choose a different one.', 'danger')
            return render_template('register.html', form=form)
        
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered. Please use a different email.', 'danger')
            return render_template('register.html', form=form)
        
        # Create new user
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch user's test history
    test_sessions = TestSession.query.filter_by(user_id=current_user.id, consent=True).order_by(TestSession.end_time.desc()).all()
    return render_template(
        'dashboard.html', 
        test_sessions=test_sessions,
        CLUSTER_DESCRIPTIONS=CLUSTER_DESCRIPTIONS
    )

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

# Routes for personality test
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/start', methods=['POST'])
def start_test():
    # Record screen dimensions
    screen_width = request.form.get('screen_width', None)
    screen_height = request.form.get('screen_height', None)
    
    # Create a new test session
    session_id = str(uuid.uuid4())
    test_session = TestSession(
        id=session_id,
        user_id=current_user.id if current_user.is_authenticated else None,
        screen_width=screen_width,
        screen_height=screen_height,
        intro_elapsed=time.time() - float(request.form.get('page_load_time', time.time())),
        ip_address=request.remote_addr
    )
    db.session.add(test_session)
    db.session.commit()
    
    # Store session ID in Flask session
    session['test_session_id'] = session_id
    session['page'] = 0
    session['start_time'] = time.time()
    session['last_answer_time'] = time.time()
    
    return redirect(url_for('question_page'))

@app.route('/questions', methods=['GET', 'POST'])
def question_page():
    # Check if session exists
    if 'test_session_id' not in session:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        # Process answers from previous page
        test_session_id = session.get('test_session_id')
        last_answer_time = session.get('last_answer_time', time.time())
        
        for key, value in request.form.items():
            if key.startswith('q_'):
                question_code = key[2:]  # Remove q_ prefix
                response_time = int((time.time() - last_answer_time) * 1000)  # Convert to ms
                
                # Save response
                response = Response(
                    session_id=test_session_id,
                    question_code=question_code,
                    answer=int(value),
                    response_time_ms=response_time
                )
                db.session.add(response)
                last_answer_time = time.time()
        
        db.session.commit()
        session['last_answer_time'] = time.time()
        
        # Move to next page
        session['page'] += 1
    
    # Determine which questions to show
    all_questions = []
    for trait in QUESTIONS:
        for i, question in enumerate(QUESTIONS[trait], 1):
            all_questions.append((f"{trait}{i}", trait, question))
    
    current_page = session.get('page', 0)
    start_idx = current_page * QUESTIONS_PER_PAGE
    end_idx = start_idx + QUESTIONS_PER_PAGE
    
    # Check if we're at the end
    if start_idx >= len(all_questions):
        return redirect(url_for('finish'))
    
    page_questions = all_questions[start_idx:min(end_idx, len(all_questions))]
    
    return render_template(
        'questions.html',
        questions=page_questions,
        current_page=current_page,
        total_pages=(len(all_questions) // QUESTIONS_PER_PAGE) + (1 if len(all_questions) % QUESTIONS_PER_PAGE > 0 else 0),
        progress=min(100, (current_page * QUESTIONS_PER_PAGE * 100) // len(all_questions))
    )

@app.route('/finish', methods=['GET', 'POST'])
def finish():
    if 'test_session_id' not in session:
        return redirect(url_for('home'))
    
    test_session_id = session.get('test_session_id')
    test_session = TestSession.query.get(test_session_id)
    
    if request.method == 'POST':
        # Record consent and finalize test
        consent = request.form.get('consent') == 'yes'
        test_session.consent = consent
        test_session.end_time = datetime.utcnow()
        test_session.test_elapsed = time.time() - session.get('start_time', time.time())
        test_session.end_elapsed = time.time() - session.get('last_answer_time', time.time())
        
        if consent:
            ordered_answers = get_ordered_responses(test_session_id)
            cluster = assign_to_cluster(ordered_answers)
            test_session.cluster_result = cluster
            
        db.session.commit()
        
        if consent:
            return redirect(url_for('results', session_id=test_session_id))
        else:
            # Clear session data
            session.pop('test_session_id', None)
            session.pop('page', None)
            session.pop('start_time', None)
            session.pop('last_answer_time', None)
            return redirect(url_for('home'))
    
    return render_template('finish.html')

def calculate_trait_scores(session_id):
    """Calculate the Big Five trait scores for a given test session."""
    responses = Response.query.filter_by(session_id=session_id).all()
    
    scores = {trait: [] for trait in QUESTIONS}
    
    for response in responses:
        trait = response.question_code[:3]
        question_num = int(response.question_code[3:])
        
        # Some questions are reverse-scored
        if trait == 'EXT' and question_num in [2, 4, 6, 8, 10]:
            scores[trait].append(6 - response.answer)  # Reverse score
        elif trait == 'EST' and question_num in [1, 3, 5, 6, 7, 8, 9, 10]:
            scores[trait].append(6 - response.answer)  # Reverse score
        elif trait == 'AGR' and question_num in [1, 3, 5, 7]:
            scores[trait].append(6 - response.answer)  # Reverse score
        elif trait == 'CSN' and question_num in [2, 4, 6, 8]:
            scores[trait].append(6 - response.answer)  # Reverse score
        elif trait == 'OPN' and question_num in [2, 4, 6]:
            scores[trait].append(6 - response.answer)  # Reverse score
        else:
            scores[trait].append(response.answer)
    
    # Calculate average scores
    avg_scores = {}
    for trait, values in scores.items():
        if values:
            avg_scores[trait] = sum(values) / len(values)
        else:
            avg_scores[trait] = 0
    
    return avg_scores

def get_ordered_responses(session_id):
    """
    Returns a list of 50 answers in the order used for model training.
    """
    responses = Response.query.filter_by(session_id=session_id).all()
    resp_dict = {r.question_code: r.answer for r in responses}
    ordered_codes = []
    for trait in ['EXT', 'EST', 'AGR', 'CSN', 'OPN']:
        for i in range(1, 11):
            ordered_codes.append(f"{trait}{i}")
    ordered_answers = [resp_dict.get(code, 3) for code in ordered_codes]  # default to 3 if missing
    return ordered_answers

def assign_to_cluster(responses):
    """
    Assign a user to a personality cluster using the trained KMeans model.
    Expects a list of 50 answers, ordered as in model training.
    """
    cluster = kmeans_model.predict([responses])[0]
    return int(cluster)

@app.route('/results/<session_id>')
def results(session_id):
    # Get test session and check consent
    test_session = TestSession.query.get(session_id)
    if not test_session or not test_session.consent:
        flash('Test results not found or not available', 'danger')
        return redirect(url_for('home'))
    
    # Check if user has permission to view these results
    if test_session.user_id and test_session.user_id != getattr(current_user, 'id', None) and not current_user.is_authenticated:
        flash('You do not have permission to view these results', 'danger')
        return redirect(url_for('login'))
    
    # Calculate trait scores
    trait_scores = calculate_trait_scores(session_id)
    
    # Get cluster results
    cluster = test_session.cluster_result
    cluster_info = CLUSTER_DESCRIPTIONS.get(cluster, {
        "name": "Uncategorized",
        "description": "Your personality profile doesn't fit neatly into our existing categories."
    })
    
    return render_template(
        'results.html',
        scores=trait_scores,
        trait_names=TRAIT_NAMES,
        cluster=cluster_info,
        test_session=test_session
    )

@app.route('/test_history')
@login_required
def test_history():
    # Fetch user's test history
    test_sessions = TestSession.query.filter_by(user_id=current_user.id, consent=True).order_by(TestSession.end_time.desc()).all()
    return render_template(
        'test_history.html', 
        test_sessions=test_sessions,
        CLUSTER_DESCRIPTIONS=CLUSTER_DESCRIPTIONS,  # Add this line
        calculate_trait_scores=calculate_trait_scores  # Also add this if you're using this function in the template
    )


# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

# Create database
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
