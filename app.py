from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
import csv
import time
import pandas as pd
from werkzeug.utils import secure_filename
import sys
import threading

# Import the existing test automation script functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_automation_script import (
    call_chatgpt_api, call_deepseek_api, is_similar, prepare_test_files,
    INPUT_DATA_PATH, OUTPUT_DATA_PATH, TEST_CATEGORIES
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store test results and progress
current_tests = []
test_results = {}
test_in_progress = False
current_test_index = 0
current_category = ""

@app.route('/')
def home():
    return render_template('index.html', categories=TEST_CATEGORIES)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the file - add it to test_data directory
        category = request.form.get('category', 'custom')
        target_path = os.path.join(INPUT_DATA_PATH, f"{category}_input.csv")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Copy the file
        import shutil
        shutil.copy(file_path, target_path)
        
        return jsonify({'success': True, 'message': f'File uploaded and processed for category: {category}'})
    
    return jsonify({'error': 'Invalid file format'})

@app.route('/start_tests', methods=['POST'])
def start_tests():
    global current_tests, test_results, test_in_progress, current_test_index, current_category
    
    # Reset test state
    current_tests = []
    test_results = {}
    current_test_index = 0
    test_in_progress = True
    
    # Get selected category
    data = request.get_json()
    category = data.get('category')
    current_category = category
    
    if not category or category not in TEST_CATEGORIES:
        return jsonify({'error': 'Invalid category'})
    
    # Load test cases from CSV
    input_file = f"{INPUT_DATA_PATH}{category}_input.csv"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                current_tests.append(row)
    except FileNotFoundError:
        prepare_test_files()  # Create template files
        return jsonify({'error': f'Test file for {category} not found. Template created, please add test cases.'})
    
    # Start testing in background thread
    test_thread = threading.Thread(target=run_tests_background)
    test_thread.daemon = True
    test_thread.start()
    
    return jsonify({'success': True, 'totalTests': len(current_tests)})

def run_tests_background():
    global current_tests, test_results, test_in_progress, current_test_index
    
    print(f"Starting background tests with {len(current_tests)} test cases")
    
    for i, test_case in enumerate(current_tests):
        current_test_index = i
        test_id = test_case.get('Test ID', f'Unknown-{i}')
        question = test_case.get('Question', '')
        expected_output = test_case.get('Expected Output', '')
        
        print(f"Processing test {i+1}/{len(current_tests)}: {test_id}")
        
        if not question:
            print(f"Skipping test {test_id} - no question")
            continue
        
        try:
            # Call APIs
            print(f"Calling ChatGPT API for test {test_id}")
            chatgpt_output = call_chatgpt_api(question)
            
            print(f"Calling DeepSeek API for test {test_id}")
            deepseek_output = call_deepseek_api(question)
            
            # Evaluate results
            print(f"Evaluating results for test {test_id}")
            chatgpt_result, chatgpt_similarity = is_similar(expected_output, chatgpt_output)
            deepseek_result, deepseek_similarity = is_similar(expected_output, deepseek_output, threshold=0.33)
            
            # Store results
            print(f"Storing results for test {test_id}")
            test_results[test_id] = {
                'question': question,
                'expected': expected_output,
                'chatgpt_output': chatgpt_output,
                'deepseek_output': deepseek_output,
                'chatgpt_result': 'Correct' if chatgpt_result else 'Incorrect',
                'deepseek_result': 'Correct' if deepseek_result else 'Incorrect',
                'chatgpt_similarity': chatgpt_similarity,
                'deepseek_similarity': deepseek_similarity
            }
            
            # Add to CSV results
            output_file = f"{OUTPUT_DATA_PATH}{current_category}_results.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if i == 0:
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        'Test ID', 'Question', 'Expected Output', 
                        'ChatGPT Output', 'DeepSeek Output', 
                        'ChatGPT Result', 'DeepSeek Result',
                        'ChatGPT Similarity', 'DeepSeek Similarity'
                    ])
            
            # Append result to CSV
            with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    test_id, question, expected_output,
                    chatgpt_output, deepseek_output,
                    'Pass' if chatgpt_result else 'Fail',
                    'Pass' if deepseek_result else 'Fail',
                    f"{chatgpt_similarity:.4f}", f"{deepseek_similarity:.4f}"
                ])
            
            print(f"Completed test {i+1}/{len(current_tests)}")
            
        except Exception as e:
            print(f"Error processing test {test_id}: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Pause between tests to avoid rate limiting
        time.sleep(1)
    
    print("All tests completed")
    test_in_progress = False

@app.route('/test_status')
def test_status():
    if not test_in_progress and current_test_index == 0 and not test_results:
        return jsonify({
            'inProgress': False,
            'completed': False,
            'currentIndex': 0,
            'totalTests': 0,
            'message': 'No tests have been started'
        })
    
    # Calculate if tests are completed
    tests_completed = not test_in_progress and current_tests and current_test_index >= len(current_tests) - 1
    
    return jsonify({
        'inProgress': test_in_progress,
        'completed': tests_completed,
        'currentIndex': current_test_index,
        'totalTests': len(current_tests),
        'currentTestId': current_tests[current_test_index].get('Test ID', f'Unknown-{current_test_index}') if current_tests else None
    })

@app.route('/current_test')
def current_test():
    if not current_tests or current_test_index >= len(current_tests):
        return jsonify({'error': 'No current test'})
    
    test_id = current_tests[current_test_index].get('Test ID', f'Unknown-{current_test_index}')
    
    # If test is still running, return just the question and expected output
    if test_id not in test_results:
        return jsonify({
            'id': test_id,
            'question': current_tests[current_test_index].get('Question', ''),
            'expected': current_tests[current_test_index].get('Expected Output', ''),
            'processing': True
        })
    
    # If test has results, return full data
    return jsonify({
        'id': test_id,
        'question': current_tests[current_test_index].get('Question', ''),
        'expected': current_tests[current_test_index].get('Expected Output', ''),
        'chatgpt_output': test_results[test_id]['chatgpt_output'],
        'deepseek_output': test_results[test_id]['deepseek_output'],
        'chatgpt_result': test_results[test_id]['chatgpt_result'],
        'deepseek_result': test_results[test_id]['deepseek_result'],
        'chatgpt_similarity': test_results[test_id]['chatgpt_similarity'],
        'deepseek_similarity': test_results[test_id]['deepseek_similarity'],
        'processing': False
    })

@app.route('/test_results')
def get_test_results():
    return jsonify(test_results)

@app.route('/summary')
def summary():
    if not test_results:
        return jsonify({'error': 'No test results available'})
    
    total_tests = len(test_results)
    chatgpt_correct = sum(1 for result in test_results.values() if result['chatgpt_result'] == 'Correct')
    deepseek_correct = sum(1 for result in test_results.values() if result['deepseek_result'] == 'Correct')
    
    return jsonify({
        'totalTests': total_tests,
        'chatgptCorrect': chatgpt_correct,
        'chatgptRate': (chatgpt_correct / total_tests) * 100 if total_tests > 0 else 0,
        'deepseekCorrect': deepseek_correct,
        'deepseekRate': (deepseek_correct / total_tests) * 100 if total_tests > 0 else 0
    })

if __name__ == '__main__':
    # Ensure test data directories exist
    os.makedirs(INPUT_DATA_PATH, exist_ok=True)
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    
    prepare_test_files()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)