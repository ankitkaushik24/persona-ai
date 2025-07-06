document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('question-form');
    const questionInput = document.getElementById('question-input');
    const answerDiv = document.getElementById('answer');

    // Set initial state
    answerDiv.textContent = 'Your answer will appear here.';
    answerDiv.classList.remove('loading');

    form.addEventListener('submit', async (event) => {
        event.preventDefault();

        const question = questionInput.value.trim();
        if (!question) {
            alert('Please enter a question.');
            return;
        }

        // Show loading state
        answerDiv.textContent = '';
        answerDiv.classList.add('loading');

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'An unknown error occurred.');
            }

            const data = await response.json();
            answerDiv.textContent = data.answer;

        } catch (error) {
            answerDiv.textContent = `Error: ${error.message}`;
        } finally {
            // Remove loading state
            answerDiv.classList.remove('loading');
        }
    });
});
