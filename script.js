document.addEventListener('DOMContentLoaded', () => {
    const startTestBtn = document.getElementById('startTestBtn');
    const categorySelect = document.getElementById('categorySelect');
    const testProgress = document.getElementById('testProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const currentCase = document.getElementById('currentCase');
    const currentQuestion = document.getElementById('currentQuestion');
    const workArea = document.getElementById('workArea');
    const expectedAnswer = document.getElementById('expectedAnswer');
    const chatgptResponse = document.getElementById('chatgptResponse');
    const deepseekResponse = document.getElementById('deepseekResponse');
    const chatgptResult = document.getElementById('chatgptResult');
    const deepseekResult = document.getElementById('deepseekResult');
    const chatgptLoader = document.getElementById('chatgptLoader');
    const deepseekLoader = document.getElementById('deepseekLoader');
    const summaryResults = document.getElementById('summaryResults');
    const chatgptPassRate = document.getElementById('chatgptPassRate');
    const deepseekPassRate = document.getElementById('deepseekPassRate');
    const chatgptPassCount = document.getElementById('chatgptPassCount');
    const deepseekPassCount = document.getElementById('deepseekPassCount');

    // Variables
    let statusInterval;
    let totalTests = 0;
    let currentIndex = 0;
    let pollDelay = 1000; // 1 second default

    // Start testing
    startTestBtn.addEventListener('click', () => {
        const category = categorySelect.value;

        // Reset UI
        testProgress.classList.remove('d-none');
        currentCase.classList.remove('d-none');
        summaryResults.classList.add('d-none');

        chatgptResponse.innerHTML = '';
        deepseekResponse.innerHTML = '';
        chatgptResult.classList.add('d-none');
        deepseekResult.classList.add('d-none');

        chatgptLoader.classList.remove('d-none');
        deepseekLoader.classList.remove('d-none');

        progressBar.style.width = '0%';
        progressText.textContent = 'Starting tests...';

        // Call backend to start tests
        fetch('/start_tests', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ category }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }

                totalTests = data.totalTests;
                currentIndex = 0;

                // Start polling for test status
                statusInterval = setInterval(checkTestStatus, pollDelay);
            })
            .catch(error => {
                console.error('Error starting tests:', error);
                alert('Error starting tests. Check console for details.');
            });
    });

    // Check test status
    function checkTestStatus() {
        fetch('/test_status')
            .then(response => response.json())
            .then(data => {
                // Update progressd
                let progress = 0;
                let progressMessage = '';

                if (data.completed) {
                    // If tests are complete, show 100% regardless of index
                    progress = 100;
                    progressMessage = `${data.totalTests} of ${data.totalTests} tests completed`;
                } else {
                    // Otherwise show current progress
                    progress = ((data.currentIndex + 1) / data.totalTests) * 100;
                    progressMessage = `${data.currentIndex + 1} of ${data.totalTests} tests completed`;
                }

                progressBar.style.width = `${progress}%`;
                progressText.textContent = progressMessage;

                // If test is still running or index changed, update current test
                if (data.inProgress || currentIndex !== data.currentIndex) {
                    currentIndex = data.currentIndex;
                    updateCurrentTest();
                }

                // If all tests are done, show summary
                if (!data.inProgress && data.completed) {
                    clearInterval(statusInterval);
                    loadSummary();
                }
            })
            .catch(error => {
                console.error('Error checking test status:', error);
            });
    }

    // Update current test display
    function updateCurrentTest() {
        fetch('/current_test')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error(data.error);
                    return;
                }

                // Update question and expected output
                currentQuestion.textContent = data.question;
                expectedAnswer.textContent = data.expected;

                if (data.question.includes('math') || data.question.includes('equation') ||
                    data.question.includes('solve')) {
                    let mathWork = '';
                    const numbers = data.question.match(/\d+/g);
                    if (numbers && numbers.length >= 2) {
                        mathWork = numbers.slice(0, 3).join(' * ') + ' = ' +
                            (numbers.slice(0, 3).reduce((a, b) => a * b, 1)) + '\n';
                    }
                    workArea.textContent = mathWork;
                    workArea.classList.remove('d-none');
                } else {
                    workArea.classList.add('d-none');
                }

                // If processing is done, show responses
                if (!data.processing) {
                    // Update ChatGPT
                    chatgptLoader.classList.add('d-none');
                    chatgptResponse.textContent = data.chatgpt_output;
                    chatgptResult.textContent = data.chatgpt_result;
                    chatgptResult.classList.remove('d-none');
                    if (data.chatgpt_result === 'Correct') {
                        chatgptResult.classList.remove('bg-danger');
                        chatgptResult.classList.add('bg-success');
                    } else {
                        chatgptResult.classList.remove('bg-success');
                        chatgptResult.classList.add('bg-danger');
                    }

                    // Update DeepSeek
                    deepseekLoader.classList.add('d-none');
                    deepseekResponse.textContent = data.deepseek_output;
                    deepseekResult.textContent = data.deepseek_result;
                    deepseekResult.classList.remove('d-none');
                    if (data.deepseek_result === 'Correct') {
                        deepseekResult.classList.remove('bg-danger');
                        deepseekResult.classList.add('bg-success');
                    } else {
                        deepseekResult.classList.remove('bg-success');
                        deepseekResult.classList.add('bg-danger');
                    }
                }
            })
            .catch(error => {
                console.error('Error getting current test:', error);
            });
    }

    // Load and display summary
    function loadSummary() {
        fetch('/summary')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error(data.error);
                    return;
                }

                // Show summary panel
                summaryResults.classList.remove('d-none');

                // Update summary data
                chatgptPassRate.textContent = `${data.chatgptRate.toFixed(1)}%`;
                deepseekPassRate.textContent = `${data.deepseekRate.toFixed(1)}%`;
                chatgptPassCount.textContent = `${data.chatgptCorrect} / ${data.totalTests} correct`;
                deepseekPassCount.textContent = `${data.deepseekCorrect} / ${data.totalTests} correct`;
            })
            .catch(error => {
                console.error('Error loading summary:', error);
            });
    }
});