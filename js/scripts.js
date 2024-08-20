// Function to position the context menu within the viewport and adjust position if necessary
function positionContextMenu(contextMenu, x, y) {
    const menuWidth = contextMenu.offsetWidth;
    const menuHeight = contextMenu.offsetHeight;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Adjust position if the menu is out of viewport
    const adjustedX = (x + menuWidth > viewportWidth) ? viewportWidth - menuWidth - 10 : x;
    const adjustedY = (y + menuHeight > viewportHeight) ? viewportHeight - menuHeight - 10 : y;

    contextMenu.style.left = `${adjustedX}px`;
    contextMenu.style.top = `${adjustedY}px`;
}

// Show the context menu when text is selected
document.addEventListener('mouseup', function (event) {
    const selectedText = window.getSelection().toString().trim();
    const contextMenu = document.getElementById('context-menu');

    if (selectedText) {
        // Show the context menu
        contextMenu.style.display = 'block';

        // If you want the menu to appear near the cursor (optional)
        // positionContextMenu(contextMenu, event.pageX, event.pageY);
    } else {
        contextMenu.style.display = 'none';
    }
});

// Handle ELI5 button click
document.getElementById('eli5-button').addEventListener('click', function () {
    const selectedText = window.getSelection().toString().trim();
    if (selectedText) {
        fetch('/eli5', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: selectedText })
        })
        .then(response => response.json())
        .then(data => {
            alert(`Response: ${data.explanation}`);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    // Hide the context menu after processing
    document.getElementById('context-menu').style.display = 'none';
});

// Hide the context menu if clicked elsewhere
document.addEventListener('click', function (event) {
    const contextMenu = document.getElementById('context-menu');
    if (!contextMenu.contains(event.target) && window.getSelection().toString().trim() === '') {
        contextMenu.style.display = 'none';
    }
});