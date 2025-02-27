document.addEventListener('DOMContentLoaded', function () {
    let isScrolling = false;

    window.addEventListener('scroll', function () {
        if (!isScrolling) {
            window.requestAnimationFrame(function () {
                const backToTopButton = document.getElementById('back-to-top');
                if (window.scrollY > 200) {
                    backToTopButton.classList.add('show');
                } else {
                    backToTopButton.classList.remove('show');
                }
                isScrolling = false;
            });
            isScrolling = true;
        }
    });
    // Check if URL contains #rss-tab
    if (window.location.hash === '#rss-tab') {
        // Find and click the RSS tab
        const rssTab = document.querySelector('a[href="#rss-tab"]');
        if (rssTab) {
            rssTab.click();
        }
    }
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
    // Get the button element
    const backToTopButton = document.getElementById('back-to-top');
    // Show or hide the button based on scroll position
    window.addEventListener('scroll', function () {
        if (window.scrollY > 200) { // Show button after scrolling 200px
            backToTopButton.classList.add('show');
        } else {
            backToTopButton.classList.remove('show');
        }
    });

    // Smooth scroll to the top when the button is clicked
    backToTopButton.addEventListener('click', function () {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
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
    // Function to filter out papers
    // Get all paper cards
    const paperCards = document.querySelectorAll('.card');

    // Function to filter papers
    function debounce(func, delay) {
        let timer;
        return function (...args) {
            clearTimeout(timer);
            timer = setTimeout(() => func.apply(this, args), delay);
        };
    }
    function addRssFeed() {
        const rssUrl = document.getElementById('rss-url').value.trim();
        if (!rssUrl) {
            alert('Please enter an RSS feed URL');
            return;
        }

        // Validate the URL format
        const urlPattern = /^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$/;
        if (!urlPattern.test(rssUrl)) {
            alert('Please enter a valid URL');
            return;
        }

        // Add the RSS feed to your list
        // Here, you would typically make an API call or update your data structure
        // For this example, we'll just show a success message
        alert('RSS feed added successfully!');

        // Clear the input
        document.getElementById('rss-url').value = '';
    }

    // Optional: Add form validation and error handling
    document.getElementById('rss-url').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            addRssFeed();
        }
    });

    function filterPapers(query) {
        query = query.toLowerCase();

        let visibleCards = 0;
        const paperContainer = document.querySelector(".row"); // Adjust if necessary

        // Filter paper cards (existing functionality)
        const paperCards = document.querySelectorAll(".col-md-6.col-lg-4"); // Select paper cards
        paperCards.forEach(card => {
            const title = card.querySelector('.card-title')?.textContent.toLowerCase() || "";
            const tags = card.querySelector('.card-footer')?.textContent.toLowerCase() || "";

            const shouldShow = title.includes(query) || tags.includes(query);

            if (shouldShow) {
                card.classList.remove("d-none"); // Bootstrap class for proper hiding
                visibleCards++;
            } else {
                card.classList.add("d-none");
            }
        });

        // Add RSS article filtering
        const rssCards = document.querySelectorAll(".rss-article-card");
        rssCards.forEach(card => {
            const title = card.querySelector('.card-title')?.textContent.toLowerCase() || "";
            const description = card.querySelector('.card-text')?.textContent.toLowerCase() || "";
            const feedName = card.querySelector('.feed-name')?.textContent.toLowerCase() || "";

            const shouldShow = title.includes(query) ||
                description.includes(query) ||
                feedName.includes(query);

            if (shouldShow) {
                card.classList.remove("d-none");
                visibleCards++;
            } else {
                card.classList.add("d-none");
            }
        });

        // If no cards match, show the alert message
        const noResultsAlert = document.querySelector(".alert-info");
        if (noResultsAlert) {
            noResultsAlert.style.display = visibleCards === 0 ? "block" : "none";
        }
    }

    // Add event listener with debounce for better performance
    document.getElementById('searchInput').addEventListener('input', debounce(function (e) {
        filterPapers(e.target.value);
    }, 150));

    // Initial filter (show all papers)
    filterPapers('');



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
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: question })
            })
                .then(response => {
                    if (!response.ok) throw new Error('Network response was not ok');
                    return response.body.getReader();
                })
                .then(reader => {
                    loadingMessage.style.display = 'none';

                    // Create a new Q&A entry
                    const newEntry = document.createElement('div');
                    newEntry.className = 'ollama-history-entry mt-3 p-2 border rounded';
                    newEntry.innerHTML = `<strong>Q:</strong> ${question}<br><strong>A:</strong> <span id="answer-text"></span>`;
                    historyContainer.appendChild(newEntry);
                    newEntry.scrollIntoView({ behavior: 'smooth', block: 'start' });

                    const answerText = newEntry.querySelector('#answer-text');
                    const decoder = new TextDecoder();
                    let accumulatedText = '';

                    function readStream() {
                        reader.read().then(({ done, value }) => {
                            if (done) return;
                            const chunk = decoder.decode(value, { stream: true });

                            // Append text with proper spacing
                            accumulatedText += chunk + ' ';

                            // Format new lines for better readability
                            answerText.innerHTML = accumulatedText
                                .replace(/\n\n/g, '<p></p>')  // Paragraphs for double new lines
                                .replace(/\n/g, ' ');         // Prevent column-like structure

                            readStream();
                        });
                    }

                    readStream();
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

        status.textContent = 'Uploading and Initializing In Memory Vector Store for RAG....'; // Show uploading status

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
    document.getElementById('rag-form').addEventListener('submit', function (event) {
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