from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
from nltk.tokenize import sent_tokenize
import nltk
import os
import fitz  # PyMuPDF for PDFs
from docx import Document  # python-docx for DOCX files
import re
from collections import Counter
import spacy
from textblob import TextBlob
import logging
from werkzeug.utils import secure_filename

# Download required NLTK data
nltk_data_dir = os.getenv("NLTK_DATA", "nltk_data")
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('maxent_ne_chunker', download_dir=nltk_data_dir)
nltk.download('words', download_dir=nltk_data_dir)

# Initialize spaCy
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the T5 model and tokenizer
model_name = "t5-base"  # Upgraded from t5-small for better quality
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Initialize sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis")

class TextAnalyzer:
    @staticmethod
    def get_reading_time(text):
        words = len(text.split())
        return round(words / 200)  # Average reading speed of 200 WPM
    
    @staticmethod
    def get_text_statistics(text):
        words = len(text.split())
        sentences = len(sent_tokenize(text))
        characters = len(text)
        paragraphs = len(text.split('\n\n'))
        
        # Calculate reading level (Flesch Reading Ease)
        blob = TextBlob(text)
        reading_ease = round(206.835 - 1.015 * (words/sentences) - 84.6 * (sum(len(word) for word in text.split())/words), 2)
        
        return {
            'word_count': words,
            'sentence_count': sentences,
            'character_count': characters,
            'paragraph_count': paragraphs,
            'reading_time': f"{round(words/200)} minutes",
            'reading_ease': reading_ease
        }
    
    @staticmethod
    def extract_keywords(text, num_keywords=5):
        doc = nlp(text)
        words = [token.text for token in doc if not token.is_stop and token.is_alpha]
        word_freq = Counter(words)
        return [word for word, count in word_freq.most_common(num_keywords)]

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        input_text = request.form.get('text', '')
        length = request.form.get('length', 'short')
        style = request.form.get('style', 'paragraph')
        include_stats = request.form.get('include_stats', 'false') == 'true'

        # Handle file upload
        if not input_text and 'fileUpload' in request.files:
            file = request.files['fileUpload']
            if file and file.filename:
                input_text = extract_text_from_file(file)

        if not input_text:
            return jsonify({'error': 'No input text provided'}), 400

        # Clean the input text
        input_text = clean_text(input_text)
        
        # Get text statistics if requested
        stats = TextAnalyzer.get_text_statistics(input_text) if include_stats else None
        
        # Generate summary
        summary = summarize_text(input_text, length, style)
        
        # Get keywords
        keywords = TextAnalyzer.extract_keywords(input_text)
        
        # Analyze sentiment
        sentiment = analyze_sentiment(summary)
        
        response = {
            'summary': summary,
            'keywords': keywords,
            'sentiment': sentiment,
            'statistics': stats
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return jsonify({'error': 'An error occurred during summarization'}), 500

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return {
        'label': result['label'],
        'score': round(result['score'], 2)
    }

def extract_text_from_file(file):
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError("Unsupported file format")
        
        # Clean up the file after extraction
        os.remove(file_path)
        return text
    
    except Exception as e:
        logger.error(f"Error in file extraction: {str(e)}")
        raise

def extract_text_from_pdf(file_path):
    with fitz.open(file_path) as pdf_document:
        text = ""
        for page in pdf_document:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def summarize_text(input_text, length, style):
    # Configure summarization parameters based on length
    length_config = {
        'short': {'max_len': 60, 'min_len': 30},
        'medium': {'max_len': 120, 'min_len': 60},
        'long': {'max_len': 200, 'min_len': 100}
    }[length]

    # Prepare input text
    input_text = f"summarize: {input_text}"
    
    # Tokenize and generate summary
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    summary_ids = model.generate(
        input_ids,
        max_length=length_config['max_len'],
        min_length=length_config['min_len'],
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Apply formatting based on style
    if style == 'bullet':
        summary = format_bullet_points(summary)
    elif style == 'outline':
        summary = format_outline(summary)
    else:
        summary = summary  # Paragraph format
        
    return summary

def format_bullet_points(text):
    sentences = sent_tokenize(text)
    return "\n• " + "\n• ".join(sentences)

def format_outline(text):
    sentences = sent_tokenize(text)
    outline = "I. Main Points\n"
    for i, sentence in enumerate(sentences, 1):
        outline += f"   {chr(64+i)}. {sentence}\n"
    return outline

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large. Maximum size is 16MB'}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)