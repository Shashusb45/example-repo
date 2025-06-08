import os
import re
import pickle
import subprocess
import pandas as pd
import logging
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, render_template, request, flash, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import ssl
from googletrans import Translator
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import io
import base64

# SSL context setup
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# NLTK downloads
# try:
#     import nltk
#     nltk.download('stopwords', quiet=True)
#     nltk.download('punkt', quiet=True)
# except Exception as e:
#     logger.error(f"NLTK Download Error: {e}")

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a real secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Load ML models
try:
    model = pickle.load(open('/Users/91963/OneDrive/Desktop/FakeReviewDetection/Models/BestModel.pkl', 'rb'))
    vectorizer = pickle.load(open('/Users/91963/OneDrive/Desktop/FakeReviewDetection/Models/CountVectorizer.pkl', 'rb'))
except FileNotFoundError as e:
    logger.error(f"Model or Vectorizer not found: {e}")
    model, vectorizer = None, None

sw = set(stopwords.words('english')) if 'stopwords' in dir() else set()

# Text preprocessing
def text_preprocessing(review):
    try:
        txt = TextBlob(review)
        result = txt.correct()
        cleaned = re.sub("[^a-zA-Z]", " ", str(result)).lower().split()
        stemmed = [PorterStemmer().stem(word) for word in cleaned if word not in sw]
        return " ".join(stemmed)
    except Exception as e:
        logger.error(f"Text Preprocessing Error: {e}")
        return review

# Text classification
def text_classification(review):
    if len(review) < 1:
        print("Invalid") 
    else:
        cleaned_review = text_preprocessing(review)
        process = vectorizer.transform([cleaned_review]).toarray()
        prediction = model.predict(process)
        p = ''.join(str(i) for i in prediction)
        print("review")
        
        if p == 'True':
            return "Legitimate"
        if p == 'False':
            return "Fake"
        
def is_spam(review):
    """
    Detect whether a review is spam based on predefined rules.
    """
    try:
        # 1. Excessive repetition: Check if the review has low word diversity
        words = review.split()
        unique_words_ratio = len(set(words)) / len(words) if words else 1  # Avoid division by zero
        excessive_repetition = unique_words_ratio < 0.5  # Less than 50% unique words
        
        # 2. Contains links: Check if URLs are present
        contains_links = bool(re.search(r"http[s]?://|www\.", review))
        
        # 3. Contains spammy phrases: Common promotional terms
        spammy_phrases = [
            "buy now", "click here", "limited offer", "subscribe", 
            "free", "order now", "act fast", "best deal", "100% guarantee"
        ]
        contains_spammy_words = any(phrase in review.lower() for phrase in spammy_phrases)
        
        # 4. Excessive punctuation: Check for too many exclamation marks
        excessive_exclamations = review.count("!") > 3  # More than 3 exclamation marks
        
        # 5. Gibberish or nonsensical content: Low polarity in sentiment
        polarity = TextBlob(review).sentiment.polarity  # Sentiment polarity (-1 to 1)
        nonsensical_content = polarity == 0 and len(words) > 10  # Neutral sentiment in a long review
        
        # Combine the rules to classify as spam
        if excessive_repetition or contains_links or contains_spammy_words or excessive_exclamations or nonsensical_content:
            return "Spam"
        else:
            return "Not Spam"
    except Exception as e:
        logger.error(f"Spam Detection Error: {e}")
        return "Error"

# Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or Email already exists', 'error')
            return redirect(url_for('signup'))
        hashed_password = generate_password_hash(password)
        db.session.add(User(name=name, username=username, email=email, password=hashed_password))
        db.session.commit()
        flash('Signup successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'], session['name'] = user.id, user.name
            return redirect(url_for('index'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index_custom.html', name=session['name'])

@app.route('/review_analysis', methods=['GET', 'POST'])
def review_analysis():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    analysis_results = None
    
    if request.method == 'POST':
        review = request.form.get('review', '')
        if review:
            # Get fake/real prediction
            legitimacy = text_classification(review)
            
            # Perform sentiment analysis using TextBlob
            blob = TextBlob(review)
            sentiment_score = blob.sentiment.polarity
            
            # Determine sentiment category
            if sentiment_score > 0.2:
                sentiment = "Positive"
            elif sentiment_score < -0.2:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            # Determine mood based on subjectivity and polarity
            subjectivity = blob.sentiment.subjectivity
            if sentiment == "Positive":
                if subjectivity > 0.7:
                    mood = "Happy"
                else:
                    mood = "Satisfied"
            elif sentiment == "Negative":
                if subjectivity > 0.7:
                    mood = "Angry"
                else:
                    mood = "Sad"
            else:
                mood = "Neutral"
            
            # Create analysis results dictionary
            analysis_results = {
                'review': review,
                'legitimacy': legitimacy,
                'sentiment': sentiment,
                'sentiment_score': round(sentiment_score * 100, 1),
                'subjectivity_score': round(subjectivity * 100, 1),
                'mood': mood
            }
            
    return render_template('review_analysis.html', analysis=analysis_results)


@app.route('/review_crawler', methods=['GET', 'POST'])
def review_crawler():
    """Page to start crawling reviews and display results"""
    results = []  # To store the review results
    error_message = None
    
    if request.method == 'POST':
        url = request.form.get('url', '').strip()  # Get URL from form input
        
        if url:
            try:
                # Run review crawler script (Make sure the Python script is implemented correctly)
                result = subprocess.run(
                    f"python3 review_crawler.py {url}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=60  # Set a timeout value (in seconds)
                )

                # Log any output or errors from the subprocess
                logger.info(f"Crawler STDOUT: {result.stdout}")
                if result.stderr:
                    logger.error(f"Crawler STDERR: {result.stderr}")
                
                # Check if the crawler produced a valid reviews file
                reviews_file = 'reviews1.csv'
                if os.path.exists(reviews_file):
                    df = pd.read_csv(reviews_file)  # Read reviews from the file
                    reviews = df.get('review', []).tolist()
                    
                    # Classify each review
                    for review in reviews:
                        prediction = text_classification(review)
                        status = "Legitimate" if prediction == 'Legitimate' else "Fake"
                        results.append({
                            'review': review,
                            'status': status
                        })
                else:
                    error_message = "Crawler did not produce a valid reviews file."

            except subprocess.TimeoutExpired:
                error_message = "The review crawler took too long and was stopped."
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                error_message = f"An error occurred: {str(e)}"
    
    return render_template('review_crawler.html', results=results, error_message=error_message)



@app.route('/spam-detection', methods=['GET', 'POST'])
def spam_detection():
    """
    Spam detection route to analyze reviews for spammy content.
    """
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    result = None
    review = None

    if request.method == 'POST':
        review = request.form.get('review', '').strip()
        if review:
            result = is_spam(review)
    
    return render_template('spam_detection.html', review=review, result=result)


from deep_translator import GoogleTranslator

@app.route('/review-analysis-multi-language', methods=['GET', 'POST'])
def review_analysis_multi_language():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Dictionary of supported languages
    languages = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'hi': 'Hindi',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh-CN': 'Chinese (Simplified)',
        'ru': 'Russian',
        'ar': 'Arabic',
        'kn': 'Kannada'
    }
    
    analysis_results = None
    
    if request.method == 'POST':
        review = request.form.get('review', '').strip()
        source_lang = request.form.get('language', 'en')
        
        if review:
            try:
                # Store original text
                original_text = review
                
                # Translate to English if not already in English
                if source_lang != 'en':
                    translator = GoogleTranslator(source=source_lang, target='en')
                    translated_text = translator.translate(text=review)
                    review_to_analyze = translated_text
                else:
                    review_to_analyze = review
                    translated_text = review
                
                # Get fake/real prediction
                legitimacy = text_classification(review_to_analyze)
                
                # Perform sentiment analysis using TextBlob
                blob = TextBlob(review_to_analyze)
                sentiment_score = blob.sentiment.polarity
                
                # Determine sentiment category
                if sentiment_score > 0.2:
                    sentiment = "Positive"
                elif sentiment_score < -0.2:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                
                # Determine mood based on subjectivity and polarity
                subjectivity = blob.sentiment.subjectivity
                if sentiment == "Positive":
                    if subjectivity > 0.7:
                        mood = "Happy"
                    else:
                        mood = "Satisfied"
                elif sentiment == "Negative":
                    if subjectivity > 0.7:
                        mood = "Angry"
                    else:
                        mood = "Sad"
                else:
                    mood = "Neutral"
                
                # Create analysis results dictionary
                analysis_results = {
                    'review': review,
                    'original_text': original_text,
                    'translated_text': translated_text,
                    'source_language': languages.get(source_lang, source_lang),
                    'legitimacy': legitimacy,
                    'sentiment': sentiment,
                    'sentiment_score': round(sentiment_score * 100, 1),
                    'subjectivity_score': round(subjectivity * 100, 1),
                    'mood': mood
                }
                
            except Exception as e:
                logger.error(f"Translation/Analysis Error: {str(e)}")
                flash(f"An error occurred during analysis: {str(e)}", 'error')
    
    return render_template('review_analysis_multi.html', 
                         analysis=analysis_results, 
                         languages=languages)


# Route for the dummy eCommerce website
@app.route('/dummy_ecommerce_website')
def dummy_ecommerce_website():
    product = {
        'title': 'Samsung S23 Ultra 5G',
        'image_url': 'https://m.media-amazon.com/images/I/41GAnuY2-DL._SX300_SY300_QL70_FMwebp_.jpg',  # Placeholder image
        'reviews': [
            "I always wanted to experience a curved display flagship smartphone by Samsung, hence ignored the S24 Ultra and bought the S23 Ultra because it's the last curved display phone by Samsung.",
            "After using Samsung Galaxy S23 Ultra for quite a bit as a daily driver and coming back to Samsung after a long time after using other smaller form factor phones like ONEPLUS 9, here's what I think about this phone",
            "Good value for money. Will buy again."
        ]
    }
    return render_template('dummy_product_page.html', product=product)


if __name__ == '__main__':
    # with app.app_context():
    #     db.create_all()
    # app.run(debug=True)
    app.run(debug=True, port=5001)  # Change to a different port


    
