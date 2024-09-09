document.addEventListener('DOMContentLoaded', function () {
    let storedSelectedText = '';

    // Show context menu on right-click
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

    // Handle Enter key press in the custom question textarea
    document.getElementById('custom-question').addEventListener('keydown', function (event) {
        if (event.key === 'Enter' && !event.shiftKey) { // Prevent newline
            event.preventDefault();
            handleAskOllama();
        }
    });

    // Save content button
    document.getElementById('save-ollama-content').addEventListener('click', function () {
        saveOllamaContent();
    });

    // Clear content button
    document.getElementById('clear-ollama-content').addEventListener('click', function () {
        clearOllamaContent();
    });

    // Function to handle the question submission
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

                    // Create new Q&A entry
                    const newEntry = document.createElement('div');
                    newEntry.className = 'ollama-history-entry mt-3 p-2 border rounded';
                    newEntry.innerHTML = `<strong>Q:</strong> ${question}<br><strong>A:</strong> ${data.explanation}`;

                    // Append new entry to the history container
                    historyContainer.appendChild(newEntry);

                    // Scroll to the newly added entry
                    newEntry.scrollIntoView({ behavior: 'smooth', block: 'start' });

                    // Clear input and reset stored text
                    document.getElementById('custom-question').value = '';
                    storedSelectedText = '';
                })
                .catch(error => {
                    loadingMessage.style.display = 'none';
                    console.error('Error:', error);
                });
        }
    }

    // Function to save the content as markdown
    function saveOllamaContent() {
        const historyContainer = document.getElementById('askollama-history');
        const content = historyContainer.innerText.trim();
        const tags = document.getElementById('ollama-tags').value.trim();

        if (!tags) {
            alert('Please enter tags before saving.');
            return;
        }

        if (content) {
            let markdownContent = `Tags: ${tags}\n\n${content.replace(/\n/g, '\n\n')}`;

            // Send the markdown content to the server
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

    // Function to clear the content and tags
    function clearOllamaContent() {
        document.getElementById('askollama-history').innerHTML = '';
        document.getElementById('ollama-tags').value = '';
    }

    // Hide context menu if clicked outside of it
    document.addEventListener('click', function (event) {
        const contextMenu = document.getElementById('context-menu');
        const askOllamaContainer = document.getElementById('askollama-container');

        if (!askOllamaContainer.contains(event.target) && !contextMenu.contains(event.target)) {
            contextMenu.style.display = 'none';
        }
    });

    // Drag-and-Drop PDF Upload Area
    const dropArea = document.getElementById('drag-drop-area');
    const fileInput = document.getElementById('pdfFileInput');
    const status = document.getElementById('upload-status');

    // Prevent default behavior for dragover and drop events
    dropArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.add('bg-light');
    });

    dropArea.addEventListener('dragleave', function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.remove('bg-light');
    });

    dropArea.addEventListener('drop', function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.remove('bg-light');
        const files = e.dataTransfer.files;
        handleFileUpload(files);
    });

    dropArea.addEventListener('click', function () {
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        const files = fileInput.files;
        handleFileUpload(files);
    });

    function handleFileUpload(files) {
        if (files.length === 0) return;

        const file = files[0];
        if (file.type !== 'application/pdf') {
            status.textContent = 'Please upload a PDF file.';
            return;
        }

        const formData = new FormData();
        formData.append('pdfFile', file);

        status.textContent = 'Uploading...'; // Show uploading status

        fetch('/upload_pdf', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                status.textContent = 'File uploaded successfully!';

                // Create an iframe to render the PDF
                const iframe = document.createElement('iframe');
                iframe.src = data.file_url;
                iframe.width = '100%';
                iframe.height = '600px';

                // Append iframe to a container in the DOM
                document.getElementById('pdf-container').innerHTML = '';  // Clear previous content
                document.getElementById('pdf-container').appendChild(iframe);

                // Update text container
                const textContainer = document.getElementById('text-container');
                const textElement = document.createElement('pre');
                textElement.textContent = data.extracted_text || 'No text extracted from the PDF.';
                textContainer.innerHTML = '';  // Clear previous content
                textContainer.appendChild(textElement);
            } else {
                status.textContent = 'File upload failed: ' + data.message;
            }
        })
        .catch(error => {
            status.textContent = 'Error uploading file.';
            console.error('Error:', error);
        });
    }

    // Form submission for RAG query
    document.getElementById('rag-form').addEventListener('submit', function(event) {
        event.preventDefault();  // Prevent the default form submission behavior
        const query = document.getElementById('rag-query-input').value;

        // Handle the RAG query here (e.g., send it to the server via an API call)
        console.log("Query submitted: " + query);

        // Optionally update the UI with a response
        document.getElementById('rag-response').textContent = 'Processing query: ' + query;

        // Example of a fetch call for server-side processing
        fetch('/rag_query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Response data:', data); // Log the whole response
            const textContainer = document.getElementById('text-container');
            const textElement = document.createElement('pre');
            textElement.textContent = data.answer || 'No text extracted from the PDF.';
            textContainer.innerHTML = '';  // Clear previous content
            textContainer.appendChild(textElement);
            document.getElementById('rag-response').textContent = 'Processed query: ' + query;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});