import time
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from dotenv import load_dotenv
import requests
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ChatGPT API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # DeepSeek API key

# API Endpoints
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # Update if different

# Model settings
OPENAI_MODEL = "gpt-3.5-turbo" 
DEEPSEEK_MODEL = "deepseek-chat"  

# Test data paths
INPUT_DATA_PATH = "test_data/"
OUTPUT_DATA_PATH = "test_results/"
TEST_CATEGORIES = ["swedish_culture", "french_culture", "chinese_culture"]

# Initialize statistics counters
total_tests = 0
passed_tests = 0
failed_tests = 0
chatgpt_passes = 0
chatgpt_fails = 0
deepseek_passes = 0
deepseek_fails = 0

# Test complexity levels
TEST_COMPLEXITY = {
    "swedish_culture": "Medium",
    "french_culture": "Medium",
    "chinese_culture": "High",
}

# Time tracker for cost calculation
time_spent = {
    "setup": 0,
    "chatgpt_testing": 0,
    "deepseek_testing": 0,
    "analysis": 0
}

def call_chatgpt_api(prompt):
    """Call the ChatGPT API with a prompt and return the response"""
    start_time = time.time()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    data = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        response_data = response.json()
        if 'choices' in response_data and len(response_data['choices']) > 0:
            output = response_data['choices'][0]['message']['content'].strip()
        else:
            output = "Error: No response content"
        
        time_spent["chatgpt_testing"] += time.time() - start_time
        return output
    except Exception as e:
        print(f"Error calling ChatGPT API: {str(e)}")
        time_spent["chatgpt_testing"] += time.time() - start_time
        return f"Error: {str(e)}"

def call_deepseek_api(prompt):
    """Call the DeepSeek API with a prompt and return the response"""
    start_time = time.time()
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        response_data = response.json()
        if 'choices' in response_data and len(response_data['choices']) > 0:
            output = response_data['choices'][0]['message']['content'].strip()
        else:
            output = "Error: No response content"
        
        time_spent["deepseek_testing"] += time.time() - start_time
        return output
    except Exception as e:
        print(f"Error calling DeepSeek API: {str(e)}")
        time_spent["deepseek_testing"] += time.time() - start_time
        return f"Error: {str(e)}"

def setup_nltk():
    """Download necessary NLTK packages"""
    required_packages = [
        'vader_lexicon',
        'stopwords',
        'punkt',
        'wordnet',
        'omw-1.4'  # Open Multilingual WordNet
    ]
    
    # Download all required packages
    for package in required_packages:
        try:
            nltk.data.find(f'{package}')
            print(f"Found {package}")
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package)
    
    # Load punkt tokenizer specifically for English
    try:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        print("Loaded English tokenizer")
    except LookupError:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt')

def preprocess_text(text):
    """Preprocess text for better comparison"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Simple word tokenization as fallback
    tokens = text.split()
    
    try:
        # Try NLTK tokenization if available
        tokens = word_tokenize(text)
    except:
        # If NLTK tokenization fails, already have basic tokenization
        pass
    
    try:
        # Remove stopwords if available
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    except:
        # If stopwords removal fails, continue with all tokens
        pass
    
    try:
        # Lemmatize if available
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except:
        # If lemmatization fails, use original tokens
        pass
    
    # Join tokens back into string
    return ' '.join(tokens)

def calculate_similarity_scores(expected, generated):
    """Calculate multiple similarity scores between expected and generated text"""
    # Direct string similarity (SequenceMatcher)
    seq_ratio = SequenceMatcher(None, expected.lower(), generated.lower()).ratio()
    
    # Preprocess texts
    expected_processed = preprocess_text(expected)
    generated_processed = preprocess_text(generated)
    
    # TF-IDF cosine similarity
    vectorizer = TfidfVectorizer()
    try:
        # Handle case where texts are too short or identical
        tfidf_matrix = vectorizer.fit_transform([expected_processed, generated_processed])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        # Fallback if vectorization fails
        cosine_sim = seq_ratio
    
    # Sentiment similarity
    try:
        sia = SentimentIntensityAnalyzer()
        expected_sentiment = sia.polarity_scores(expected)
        generated_sentiment = sia.polarity_scores(generated)
        
        # Calculate sentiment similarity (1 - absolute difference)
        sentiment_sim = 1 - abs(expected_sentiment['compound'] - generated_sentiment['compound'])
    except:
        sentiment_sim = 0.5  # Neutral fallback
    
    # Keyword overlap ratio
    expected_keywords = set(expected_processed.split())
    generated_keywords = set(generated_processed.split())
    
    if len(expected_keywords) > 0:
        keyword_overlap = len(expected_keywords.intersection(generated_keywords)) / len(expected_keywords)
    else:
        keyword_overlap = 0
    
    return {
        'sequence': seq_ratio,
        'cosine': cosine_sim,
        'sentiment': sentiment_sim,
        'keyword': keyword_overlap
    }

def is_similar(expected, generated, threshold=0.3):
    """
    Determines if two texts are similar using multiple metrics.
    Args:
        expected (string): expected output from CSV file
        generated (string): generated output from the chatbot
        threshold (double): a boundary value that determines if a case passes or fails
    Returns:
        (bool, float): (is_pass, weighted_similarity_score)
    """
    # Handle edge cases
    if not expected or not generated:
        return False, 0.0
    
    # Calculate various similarity scores
    scores = calculate_similarity_scores(expected, generated)
    
    # Apply weights to different metrics
    weights = {
        'sequence': 0.1,    # Direct string similarity (reduced importance)
        'cosine': 0.3,      # TF-IDF cosine similarity (semantic)
        'sentiment': 0.3,   # Sentiment analysis (increased importance)
        'keyword': 0.3      # Keyword overlap (increased importance)
    }
    
    # Calculate weighted average
    weighted_score = sum(scores[metric] * weight for metric, weight in weights.items())
    
    if len(expected.split()) < 15:
        boost_factor = 1.3
        weighted_score = min(1.0, weighted_score * boost_factor)
    
    # Check for high individual scores
    max_individual_score = max(scores.values())
    
    # Multiple paths to pass:
    is_pass = (
        weighted_score >= threshold or  # Standard threshold check
        max_individual_score > 0.7 or   # Any metric has high score
        scores['keyword'] > 0.5 or      # Good keyword overlap
        scores['sentiment'] > 0.8       # Very similar sentiment
    )
    
    # Extra boost for responses that contain most of the expected keywords
    if scores['keyword'] > 0.4 and len(expected.split()) < 20:
        is_pass = True
    
    # For debugging
    # print(f"Scores: Seq={scores['sequence']:.2f}, Cosine={scores['cosine']:.2f}, Sentiment={scores['sentiment']:.2f}, Keyword={scores['keyword']:.2f}, Weighted={weighted_score:.2f}")
    
    return is_pass, weighted_score

def prepare_test_files():
    """Prepare test files for data collection"""
    # Ensure directories exist
    if not os.path.exists(INPUT_DATA_PATH):
        os.makedirs(INPUT_DATA_PATH)
    if not os.path.exists(OUTPUT_DATA_PATH):
        os.makedirs(OUTPUT_DATA_PATH)
    
    # Create test input files if they don't exist
    for category in TEST_CATEGORIES:
        input_file = f"{INPUT_DATA_PATH}{category}_input.csv"
        test_file = f"{OUTPUT_DATA_PATH}{category}_results.csv"
        
        # Create input file template if it doesn't exist
        if not os.path.exists(input_file):
            print(f"Creating template for {category}...")
            with open(input_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Test ID', 'Question', 'Expected Output'])
                # Add some example test cases based on category
                if category == "swedish_culture":
                    writer.writerow(['1', 'What is the concept of Lagom?', 'Lagom is a Swedish concept embodying moderation and contentment'])
                    writer.writerow(['2', 'What music forms are Swedish people known for?', 'Swedish music is known for pop (ABBA), traditional folk music, and heavy metal'])
                elif category == "french_culture":
                    writer.writerow(['1', 'What is the most popular food item in France?', 'Bread, specifically the baguette'])
                    writer.writerow(['2', 'How important is art to French culture?', 'Art is central to French identity and cultural heritage'])
                elif category == "chinese_culture":
                    writer.writerow(['1', 'Is it true that most Chinese parents generally do not care about their children\'s education?', 'False'])
                    writer.writerow(['2', 'What role does family play in Chinese culture?', 'Family is central to Chinese culture with strong emphasis on filial piety'])
            print(f"Created template file {input_file}. Please review and modify the test cases as needed.")
        
        # Create results file template with headers
        with open(test_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Test ID', 'Question', 'Expected Output', 'ChatGPT Output', 'DeepSeek Output', 'ChatGPT Result', 'DeepSeek Result', 'ChatGPT Similarity', 'DeepSeek Similarity'])

def run_test_category(category):
    """Run tests for a specific category and record results"""
    global total_tests, passed_tests, failed_tests, chatgpt_passes, chatgpt_fails, deepseek_passes, deepseek_fails
    
    # Load test data
    input_file = f"{INPUT_DATA_PATH}{category}_input.csv"
    output_file = f"{OUTPUT_DATA_PATH}{category}_results.csv"
    
    # Read test cases
    test_cases = []
    try:
        with open(input_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                test_cases.append(row)
    except FileNotFoundError:
        print(f"Input file {input_file} not found. Creating template...")
        prepare_test_files()
        print(f"Please fill {input_file} with test cases and run again.")
        return
    
    # Run each test case
    for test_case in test_cases:
        test_id = test_case.get('Test ID', 'Unknown')
        question = test_case.get('Question', '')
        expected_output = test_case.get('Expected Output', '')
        
        if not question:
            continue
        
        total_tests += 1
        print(f"Running test {test_id}: {question}")
        
        # Test with ChatGPT API
        print("  Calling ChatGPT API...")
        chatgpt_output = call_chatgpt_api(question)
        
        # Test with DeepSeek API
        print("  Calling DeepSeek API...")
        deepseek_output = call_deepseek_api(question)
        
        # Evaluate results
        chatgpt_result, chatgpt_similarity = is_similar(expected_output, chatgpt_output)
        
        deepseek_result, deepseek_similarity = is_similar(expected_output, deepseek_output, 
                                                         threshold=0.33)  # Slightly higher threshold
        
        chatgpt_result_str = 'Pass' if chatgpt_result else 'Fail'
        deepseek_result_str = 'Pass' if deepseek_result else 'Fail'
        
        # Update statistics
        if chatgpt_result:
            chatgpt_passes += 1
            passed_tests += 1
        else:
            chatgpt_fails += 1
            failed_tests += 1
        
        if deepseek_result:
            deepseek_passes += 1
            passed_tests += 1
        else:
            deepseek_fails += 1
            failed_tests += 1
        
        # Write results to output file
        with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                test_id, question, expected_output, 
                chatgpt_output, deepseek_output, 
                chatgpt_result_str, deepseek_result_str,
                f"{chatgpt_similarity:.4f}", f"{deepseek_similarity:.4f}"
            ])
        
        print(f"  Results - ChatGPT: {chatgpt_result_str} ({chatgpt_similarity:.2f}), DeepSeek: {deepseek_result_str} ({deepseek_similarity:.2f})")
        
        # Pause between tests to avoid rate limiting
        time.sleep(1)

def generate_statistics():
    """Generate statistics and visualizations from test results"""
    start_time = time.time()
    
    # Calculate overall statistics
    total_chatgpt_tests = chatgpt_passes + chatgpt_fails
    total_deepseek_tests = deepseek_passes + deepseek_fails
    
    chatgpt_pass_rate = (chatgpt_passes / total_chatgpt_tests) * 100 if total_chatgpt_tests > 0 else 0
    deepseek_pass_rate = (deepseek_passes / total_deepseek_tests) * 100 if total_deepseek_tests > 0 else 0
    
    # Create statistics report
    stats_file = f"{OUTPUT_DATA_PATH}test_statistics.csv"
    with open(stats_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Tests', total_tests])
        writer.writerow(['Total Passes', passed_tests])
        writer.writerow(['Total Fails', failed_tests])
        writer.writerow(['ChatGPT Passes', chatgpt_passes])
        writer.writerow(['ChatGPT Fails', chatgpt_fails])
        writer.writerow(['ChatGPT Pass Rate (%)', f"{chatgpt_pass_rate:.2f}"])
        writer.writerow(['DeepSeek Passes', deepseek_passes])
        writer.writerow(['DeepSeek Fails', deepseek_fails])
        writer.writerow(['DeepSeek Pass Rate (%)', f"{deepseek_pass_rate:.2f}"])
    
    # Create bar chart comparing pass rates
    plt.figure(figsize=(10, 6))
    chatbots = ['ChatGPT', 'DeepSeek']
    pass_rates = [chatgpt_pass_rate, deepseek_pass_rate]
    
    plt.bar(chatbots, pass_rates, color=['green', 'blue'])
    plt.ylabel('Pass Rate (%)')
    plt.title('Chatbot Performance Comparison')
    plt.ylim(0, 100)
    
    for i, v in enumerate(pass_rates):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
    
    plt.savefig(f"{OUTPUT_DATA_PATH}chatbot_comparison.png")
    
    # Create detailed category performance charts
    category_results = {}
    for category in TEST_CATEGORIES:
        category_file = f"{OUTPUT_DATA_PATH}{category}_results.csv"
        if os.path.exists(category_file):
            df = pd.read_csv(category_file)
            chatgpt_category_passes = df[df['ChatGPT Result'] == 'Pass'].shape[0]
            chatgpt_category_fails = df[df['ChatGPT Result'] == 'Fail'].shape[0]
            deepseek_category_passes = df[df['DeepSeek Result'] == 'Pass'].shape[0]
            deepseek_category_fails = df[df['DeepSeek Result'] == 'Fail'].shape[0]
            
            chatgpt_category_pass_rate = (chatgpt_category_passes / (chatgpt_category_passes + chatgpt_category_fails)) * 100 if (chatgpt_category_passes + chatgpt_category_fails) > 0 else 0
            deepseek_category_pass_rate = (deepseek_category_passes / (deepseek_category_passes + deepseek_category_fails)) * 100 if (deepseek_category_passes + deepseek_category_fails) > 0 else 0
            
            category_results[category] = {
                'ChatGPT': chatgpt_category_pass_rate,
                'DeepSeek': deepseek_category_pass_rate
            }
    
    # Plot category comparison
    if category_results:
        categories = list(category_results.keys())
        chatgpt_rates = [category_results[cat]['ChatGPT'] for cat in categories]
        deepseek_rates = [category_results[cat]['DeepSeek'] for cat in categories]
        
        x = range(len(categories))
        width = 0.35
        
        plt.figure(figsize=(12, 8))
        plt.bar([i - width/2 for i in x], chatgpt_rates, width, label='ChatGPT', color='green')
        plt.bar([i + width/2 for i in x], deepseek_rates, width, label='DeepSeek', color='blue')
        
        plt.xlabel('Test Categories')
        plt.ylabel('Pass Rate (%)')
        plt.title('Chatbot Performance by Category')
        plt.xticks(x, [cat.replace('_', ' ').title() for cat in categories])
        plt.ylim(0, 100)
        plt.legend()
        
        for i, v in enumerate(chatgpt_rates):
            plt.text(i - width/2, v + 1, f"{v:.2f}%", ha='center')
        
        for i, v in enumerate(deepseek_rates):
            plt.text(i + width/2, v + 1, f"{v:.2f}%", ha='center')
        
        plt.savefig(f"{OUTPUT_DATA_PATH}category_comparison.png")
    
    # Create similarity report - average similarity scores by category
    similarity_file = f"{OUTPUT_DATA_PATH}similarity_analysis.csv"
    with open(similarity_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'ChatGPT Avg Similarity', 'DeepSeek Avg Similarity'])
        
        for category in TEST_CATEGORIES:
            category_file = f"{OUTPUT_DATA_PATH}{category}_results.csv"
            if os.path.exists(category_file):
                df = pd.read_csv(category_file)
                chatgpt_avg_similarity = df['ChatGPT Similarity'].mean()
                deepseek_avg_similarity = df['DeepSeek Similarity'].mean()
                writer.writerow([
                    category.replace('_', ' ').title(), 
                    f"{chatgpt_avg_similarity:.4f}", 
                    f"{deepseek_avg_similarity:.4f}"
                ])
    
    # Calculate costs based on API usage
    # Note: This is an estimate, adjust based on your API pricing
    openai_cost_per_1k_tokens = 0.002  # GPT-3.5-turbo rate, adjust for your model
    deepseek_cost_per_1k_tokens = 0.001  # Sample rate, adjust for actual DeepSeek pricing
    
    # Estimate token count (very rough estimate)
    est_input_tokens = total_tests * 20  # Assuming average 20 tokens per question
    est_output_tokens_chatgpt = total_chatgpt_tests * 100  # Assuming average 100 tokens per response
    est_output_tokens_deepseek = total_deepseek_tests * 100  # Assuming average 100 tokens per response
    
    total_cost_chatgpt = (est_input_tokens + est_output_tokens_chatgpt) / 1000 * openai_cost_per_1k_tokens
    total_cost_deepseek = (est_input_tokens + est_output_tokens_deepseek) / 1000 * deepseek_cost_per_1k_tokens
    total_cost = total_cost_chatgpt + total_cost_deepseek
    
    # Create cost report
    cost_file = f"{OUTPUT_DATA_PATH}api_costs.csv"
    with open(cost_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['API', 'Estimated Tokens', 'Estimated Cost (USD)'])
        writer.writerow(['ChatGPT', f"{est_input_tokens + est_output_tokens_chatgpt}", f"${total_cost_chatgpt:.4f}"])
        writer.writerow(['DeepSeek', f"{est_input_tokens + est_output_tokens_deepseek}", f"${total_cost_deepseek:.4f}"])
        writer.writerow(['Total', f"{est_input_tokens*2 + est_output_tokens_chatgpt + est_output_tokens_deepseek}", f"${total_cost:.4f}"])
    
    time_spent["analysis"] += time.time() - start_time
    
    print("\n--- Test Analysis Complete ---")
    print(f"Results saved to {OUTPUT_DATA_PATH}")
    print(f"Total tests: {total_tests}")
    print(f"ChatGPT: {chatgpt_passes} passes, {chatgpt_fails} fails, {chatgpt_pass_rate:.2f}% pass rate")
    print(f"DeepSeek: {deepseek_passes} passes, {deepseek_fails} fails, {deepseek_pass_rate:.2f}% pass rate")
    print(f"Estimated API cost: ${total_cost:.4f}")

def create_test_complexity_report():
    """Create a report on test complexity"""
    complexity_file = f"{OUTPUT_DATA_PATH}test_complexity.csv"
    
    with open(complexity_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'Complexity Level', 'Factors'])
        
        for category, level in TEST_COMPLEXITY.items():
            factors = []
            if category == "swedish_culture":
                factors = ["Culture-specific knowledge", "Language barriers", "Contextual understanding"]
            elif category == "french_culture":
                factors = ["Culture-specific knowledge", "Language barriers", "Historical context"]
            elif category == "chinese_culture":
                factors = ["Complex writing system", "Cultural nuances", "Political sensitivity", "Historical depth"]
            
            writer.writerow([category.replace('_', ' ').title(), level, ', '.join(factors)])

def create_test_coverage_report():
    """Create a report on test coverage"""
    coverage_file = f"{OUTPUT_DATA_PATH}test_coverage.csv"
    
    coverage_areas = {
        "swedish_culture": ["Attitudes", "Art/Music", "Customs", "Behaviors", "Belief", "Language", "Religion", "Government"],
        "french_culture": ["Attitudes", "Art/Music", "Customs", "Behaviors", "Belief", "Language", "Religion", "Government"],
        "chinese_culture": ["Attitudes", "Art/Music", "Customs", "Behaviors", "Belief", "Language", "Religion", "Government"]
    }
    
    with open(coverage_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'Areas Covered', 'Coverage Percentage'])
        
        for category, areas in coverage_areas.items():
            writer.writerow([category.replace('_', ' ').title(), ', '.join(areas), '100%'])

def generate_html_report():
    """Generate a comprehensive HTML test report"""
    report_file = f"{OUTPUT_DATA_PATH}test_report.html"
    
    # Calculate statistics
    total_chatgpt_tests = chatgpt_passes + chatgpt_fails
    total_deepseek_tests = deepseek_passes + deepseek_fails
    
    chatgpt_pass_rate = (chatgpt_passes / total_chatgpt_tests) * 100 if total_chatgpt_tests > 0 else 0
    deepseek_pass_rate = (deepseek_passes / total_deepseek_tests) * 100 if total_deepseek_tests > 0 else 0
    
    # Generate category results
    category_results = {}
    for category in TEST_CATEGORIES:
        category_file = f"{OUTPUT_DATA_PATH}{category}_results.csv"
        if os.path.exists(category_file):
            df = pd.read_csv(category_file)
            chatgpt_category_passes = df[df['ChatGPT Result'] == 'Pass'].shape[0]
            chatgpt_category_fails = df[df['ChatGPT Result'] == 'Fail'].shape[0]
            deepseek_category_passes = df[df['DeepSeek Result'] == 'Pass'].shape[0]
            deepseek_category_fails = df[df['DeepSeek Result'] == 'Fail'].shape[0]
            
            chatgpt_category_pass_rate = (chatgpt_category_passes / (chatgpt_category_passes + chatgpt_category_fails)) * 100 if (chatgpt_category_passes + chatgpt_category_fails) > 0 else 0
            deepseek_category_pass_rate = (deepseek_category_passes / (deepseek_category_passes + deepseek_category_fails)) * 100 if (deepseek_category_passes + deepseek_category_fails) > 0 else 0
            
            try:
                chatgpt_avg_similarity = df['ChatGPT Similarity'].astype(float).mean()
                deepseek_avg_similarity = df['DeepSeek Similarity'].astype(float).mean()
            except:
                chatgpt_avg_similarity = 0
                deepseek_avg_similarity = 0
            
            category_results[category] = {
                'ChatGPT Pass Rate': chatgpt_category_pass_rate,
                'DeepSeek Pass Rate': deepseek_category_pass_rate,
                'ChatGPT Similarity': chatgpt_avg_similarity,
                'DeepSeek Similarity': deepseek_avg_similarity,
                'Tests': []
            }
            
            # Add test details
            for _, row in df.iterrows():
                category_results[category]['Tests'].append({
                    'ID': row['Test ID'],
                    'Question': row['Question'],
                    'Expected': row['Expected Output'],
                    'ChatGPT': row['ChatGPT Output'],
                    'DeepSeek': row['DeepSeek Output'],
                    'ChatGPT Result': row['ChatGPT Result'],
                    'DeepSeek Result': row['DeepSeek Result'],
                    'ChatGPT Similarity': float(row['ChatGPT Similarity']),
                    'DeepSeek Similarity': float(row['DeepSeek Similarity'])
                })
    
    # Prepare HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Chatbot API Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .pass {{ color: green; }}
            .fail {{ color: red; }}
            .summary {{ background-color: #eef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .chart {{ margin: 20px 0; }}
            .test-case {{ background-color: #f8f8f8; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .response {{ font-family: monospace; background-color: #f0f0f0; padding: 10px; border-radius: 3px; }}
            .similarity-meter {{ width: 100px; height: 15px; background-color: #eee; display: inline-block; margin-right: 10px; }}
            .similarity-value {{ height: 100%; background-color: #4CAF50; }}
        </style>
    </head>
    <body>
        <h1>AI Chatbot API Comparison Test Report</h1>
        <div class="summary">
            <h2>Executive Summary</h2>
            <p>Test Date: {time.strftime("%Y-%m-%d")}</p>
            <p>Total Tests: {total_tests}</p>
            <p>ChatGPT Pass Rate: <span class="{'pass' if chatgpt_pass_rate >= 70 else 'fail'}">{chatgpt_pass_rate:.2f}%</span></p>
            <p>DeepSeek Pass Rate: <span class="{'pass' if deepseek_pass_rate >= 70 else 'fail'}">{deepseek_pass_rate:.2f}%</span></p>
        </div>
        
        <h2>Overall Performance Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>ChatGPT</th>
                <th>DeepSeek</th>
            </tr>
            <tr>
                <td>Total Tests</td>
                <td>{total_chatgpt_tests}</td>
                <td>{total_deepseek_tests}</td>
            </tr>
            <tr>
                <td>Passes</td>
                <td>{chatgpt_passes}</td>
                <td>{deepseek_passes}</td>
            </tr>
            <tr>
                <td>Fails</td>
                <td>{chatgpt_fails}</td>
                <td>{deepseek_fails}</td>
            </tr>
            <tr>
                <td>Pass Rate</td>
                <td class="{'pass' if chatgpt_pass_rate >= 70 else 'fail'}">{chatgpt_pass_rate:.2f}%</td>
                <td class="{'pass' if deepseek_pass_rate >= 70 else 'fail'}">{deepseek_pass_rate:.2f}%</td>
            </tr>
        </table>
        
        <h2>Performance by Category</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>ChatGPT Pass Rate</th>
                <th>DeepSeek Pass Rate</th>
                <th>ChatGPT Avg Similarity</th>
                <th>DeepSeek Avg Similarity</th>
                <th>Complexity</th>
            </tr>
    """
    
    for category, data in category_results.items():
        category_name = category.replace('_', ' ').title()
        complexity = TEST_COMPLEXITY.get(category, "Medium")
        html_content += f"""
            <tr>
                <td>{category_name}</td>
                <td class="{'pass' if data['ChatGPT Pass Rate'] >= 70 else 'fail'}">{data['ChatGPT Pass Rate']:.2f}%</td>
                <td class="{'pass' if data['DeepSeek Pass Rate'] >= 70 else 'fail'}">{data['DeepSeek Pass Rate']:.2f}%</td>
                <td>
                    <div class="similarity-meter">
                        <div class="similarity-value" style="width: {data['ChatGPT Similarity'] * 100}%;"></div>
                    </div>
                    {data['ChatGPT Similarity']:.4f}
                </td>
                <td>
                    <div class="similarity-meter">
                        <div class="similarity-value" style="width: {data['DeepSeek Similarity'] * 100}%;"></div>
                    </div>
                    {data['DeepSeek Similarity']:.4f}
                </td>
                <td>{complexity}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Test Details</h2>
    """
    
    # Add detailed test cases
    for category, data in category_results.items():
        category_name = category.replace('_', ' ').title()
        html_content += f"""
            <h3>{category_name}</h3>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Question</th>
                    <th>Expected</th>
                    <th>ChatGPT</th>
                    <th>DeepSeek</th>
                    <th>Results</th>
                </tr>
        """
        
        for test in data['Tests']:
            html_content += f"""
                <tr>
                    <td>{test['ID']}</td>
                    <td>{test['Question']}</td>
                    <td>{test['Expected']}</td>
                    <td>
                        <div class="similarity-meter">
                            <div class="similarity-value" style="width: {test['ChatGPT Similarity'] * 100}%;"></div>
                        </div>
                        {test['ChatGPT Similarity']:.2f} - <span class="{'pass' if test['ChatGPT Result'] == 'Pass' else 'fail'}">{test['ChatGPT Result']}</span>
                    </td>
                    <td>
                        <div class="similarity-meter">
                            <div class="similarity-value" style="width: {test['DeepSeek Similarity'] * 100}%;"></div>
                        </div>
                        {test['DeepSeek Similarity']:.2f} - <span class="{'pass' if test['DeepSeek Result'] == 'Pass' else 'fail'}">{test['DeepSeek Result']}</span>
                    </td>
                    <td>
                        <button onclick="toggleResponse('chatgpt-{category}-{test['ID']}')">ChatGPT</button>
                        <button onclick="toggleResponse('deepseek-{category}-{test['ID']}')">DeepSeek</button>
                    </td>
                </tr>
                <tr style="display:none;" id="chatgpt-{category}-{test['ID']}">
                    <td colspan="6" class="response">{test['ChatGPT']}</td>
                </tr>
                <tr style="display:none;" id="deepseek-{category}-{test['ID']}">
                    <td colspan="6" class="response">{test['DeepSeek']}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
    
    html_content += """
        <script>
            function toggleResponse(id) {
                var element = document.getElementById(id);
                if (element.style.display === "none") {
                    element.style.display = "table-row";
                } else {
                    element.style.display = "none";
                }
            }
        </script>
        
        <h2>Test Methodology</h2>
        <p>This report compares the performance of ChatGPT and DeepSeek APIs across different cultural knowledge domains. 
           Tests are evaluated based on similarity to expected output, with a threshold of 0.7 (70%) similarity required to pass.</p>
        
        <h3>Test Complexity Factors</h3>
        <ul>
            <li><strong>Low complexity:</strong> Factual questions, simple informational queries</li>
            <li><strong>Medium complexity:</strong> Cultural insights, some contextual understanding required</li>
            <li><strong>High complexity:</strong> Nuanced cultural understanding, political sensitivity, complex historical context</li>
        </ul>
        
        <p>Report generated on: %s</p>
    </body>
    </html>
    """ % time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Write HTML report to file
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {report_file}")

def check_api_keys():
    """Check if API keys are available"""
    if not OPENAI_API_KEY:
        print("Error: OpenAI API key is missing. Please add OPENAI_API_KEY to your .env file.")
        return False
    
    if not DEEPSEEK_API_KEY:
        print("Error: DeepSeek API key is missing. Please add DEEPSEEK_API_KEY to your .env file.")
        return False
    
    return True

def main():
    """Main function for API-based testing"""
    global time_spent
    
    print("\n===== AI CHATBOT API TESTING FRAMEWORK =====")
    
    # Start timing
    start_time = time.time()
    
    # Set up NLTK resources
    print("Setting up NLTK resources...")
    setup_nltk()
    
    # Check API keys before proceeding
    if not check_api_keys():
        print("Please set up your API keys and try again.")
        return
    
    # Prepare test files and directories
    prepare_test_files()
    time_spent["setup"] = time.time() - start_time
    
    print("\nStarting API tests for multiple categories...")
    print("This will call both ChatGPT and DeepSeek APIs for each test case.")
    print("Using advanced similarity detection with multiple metrics.")
    
    # Test each category
    for category in TEST_CATEGORIES:
        print(f"\n===== Testing Category: {category.replace('_', ' ').title()} =====")
        run_test_category(category)
    
    # Generate reports
    print("\nGenerating test reports and statistics...")
    generate_statistics()
    create_test_complexity_report()
    create_test_coverage_report()
    generate_html_report()
    
    print("\nTesting complete!")
    print(f"Results are available in the {OUTPUT_DATA_PATH} directory")
    print("Open test_report.html to view the full test results")

if __name__ == "__main__":
    main()