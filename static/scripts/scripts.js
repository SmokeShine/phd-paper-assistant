document.addEventListener('DOMContentLoaded', function () {
    let storedSelectedText = '';

    document.addEventListener('contextmenu', function (event) {
        const selection = window.getSelection();
        const selectedText = selection.toString().trim();
        const contextMenu = document.getElementById('context-menu');
        const customQuestionInput = document.getElementById('custom-question');
        const selectedTextElement = document.getElementById('selected-text');

        if (selectedText) {
            event.preventDefault();
            storedSelectedText = selectedText;
            selectedTextElement.textContent = `Selected Text: "${selectedText}"`;
            contextMenu.style.display = 'block';
            customQuestionInput.focus();
        } else {
            contextMenu.style.display = 'none';
        }
    });

    document.getElementById('custom-question').addEventListener('keydown', function (event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            handleAskOllama();
        }
    });

    document.getElementById('save-ollama-content').addEventListener('click', function () {
        saveOllamaContent();
    });

    document.getElementById('clear-ollama-content').addEventListener('click', function () {
        clearOllamaContent();
    });

    function handleAskOllama() {
        const customQuestion = document.getElementById('custom-question').value.trim();
        const historyContainer = document.getElementById('askollama-history');
        const loadingMessage = document.getElementById('loading-message');
        let question;

        if (customQuestion) {
            question = `${customQuestion}: ${storedSelectedText}`;
        } else if (storedSelectedText) {
            question = `Explain: ${storedSelectedText}`;
        }

        if (question) {
            loadingMessage.style.display = 'block';

            fetch('/eli5', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: question })
            })
                .then(response => response.json())
                .then(data => {
                    loadingMessage.style.display = 'none';
                    const newEntry = document.createElement('div');
                    newEntry.className = 'ollama-history-entry mt-3 p-2 border rounded';
                    newEntry.innerHTML = `<strong>Q:</strong> ${question}<br><strong>A:</strong> ${data.explanation}`;
                    historyContainer.appendChild(newEntry);
                    document.getElementById('custom-question').value = '';
                    storedSelectedText = '';
                })
                .catch(error => {
                    loadingMessage.style.display = 'none';
                    console.error('Error:', error);
                });
        }
    }

    function saveOllamaContent() {
        const historyContainer = document.getElementById('askollama-history');
        const content = historyContainer.innerText.trim();
        const tags = document.getElementById('ollama-tags').value.trim();

        if (!tags) {
            alert('Please enter tags before saving.');
            return;
        }

        if (content) {
            let markdownContent = '';
            markdownContent += `Tags: ${tags}\n\n`;
            markdownContent += content.replace(/\n/g, '\n\n');

            fetch('/save_markdown', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ content: markdownContent, tags: tags })
            })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert(data.message);
                        clearOllamaContent();
                    } else {
                        alert('Error saving the file: ' + data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error saving the file. Please try again.');
                });
        }
    }

    function clearOllamaContent() {
        document.getElementById('askollama-history').innerHTML = '';
        document.getElementById('ollama-tags').value = '';
    }

    document.addEventListener('click', function (event) {
        const contextMenu = document.getElementById('context-menu');
        const askOllamaContainer = document.getElementById('askollama-container');

        if (!askOllamaContainer.contains(event.target) && !contextMenu.contains(event.target)) {
            contextMenu.style.display = 'none';
        }
    });
});