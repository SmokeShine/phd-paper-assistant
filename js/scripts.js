// document.addEventListener('mouseup', function () {
//     const selectedText = window.getSelection().toString().trim();
//     const contextMenu = document.getElementById('context-menu');

//     if (selectedText) {
//         // Show the context menu
//         contextMenu.style.display = 'block';
//     } else {
//         // Hide the context menu if no text is selected
//         contextMenu.style.display = 'none';
//     }
// });

// // Handle Ask Ollama button click
// document.getElementById('eli5-button').addEventListener('click', function () {
//     const selectedText = window.getSelection().toString().trim();
//     if (selectedText) {
//         fetch('/eli5', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify({ text: selectedText })
//         })
//             .then(response => response.json())
//             .then(data => {
//                 alert(`Explanation: ${data.explanation}`);
//             })
//             .catch(error => {
//                 console.error('Error:', error);
//             });
//     }
//     // Hide the context menu after processing
//     document.getElementById('context-menu').style.display = 'none';
// });

// // Hide the context menu if clicked elsewhere
// document.addEventListener('click', function (event) {
//     const contextMenu = document.getElementById('context-menu');
//     if (!contextMenu.contains(event.target) && window.getSelection().toString().trim() === '') {
//         contextMenu.style.display = 'none';
//     }
// });