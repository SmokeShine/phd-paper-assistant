import os

import requests
from flask import Flask, flash, redirect, render_template, request, session, jsonify,send_from_directory
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from cs50 import SQL
from helpers import apology, login_required, lookup_titles
import ollama
import uuid
import markdown 

# Configure application
app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
UPLOAD_FOLDER = 'upload_files'  # Directory to save files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")

TAGS_FOLDER = 'tags'
os.makedirs(TAGS_FOLDER, exist_ok=True)

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
@login_required
def index():
    """Show Hugging Face Papers"""
    papers_data = lookup_titles()
    if len(papers_data) == 0:
        return apology("No Data", 200)
    return render_template("index.html", papers_data=papers_data), 200


@app.route("/history")
@login_required
def history():
    """Show history of md saved"""
    TAGS_FOLDER = 'tags'
    saved_data = []

    if os.path.exists(TAGS_FOLDER):
        for filename in os.listdir(TAGS_FOLDER):
            if filename.endswith('.md'):
                file_path = os.path.join(TAGS_FOLDER, filename)
                with open(file_path, 'r') as file:
                    # Extract the tags from the markdown file (assuming the first line is the tags)
                    first_line = file.readline().strip()
                    if first_line.startswith('Tags:'):
                        tags = first_line.replace('Tags:', '').strip()
                    else:
                        tags = 'No tags'
                    
                    # Read the rest of the file content
                    content = file.read()

                # Convert Markdown to HTML
                html_content = markdown.markdown(content)
                
                saved_data.append({'name': filename, 'tags': tags, 'content': html_content})

    return render_template("history.html", saved_data=saved_data), 200


@app.route("/search", methods=["GET", "POST"])
@login_required
def search():
    """Search markdown files by tags"""
    TAGS_FOLDER = 'tags'
    filtered_data = []
    
    if request.method == 'POST':
        search_query = request.form.get('search_query', '').strip().lower()
        
        if search_query:
            if os.path.exists(TAGS_FOLDER):
                for filename in os.listdir(TAGS_FOLDER):
                    if filename.endswith('.md'):
                        file_path = os.path.join(TAGS_FOLDER, filename)
                        with open(file_path, 'r') as file:
                            first_line = file.readline().strip()
                            if first_line.startswith('Tags:'):
                                tags = first_line.replace('Tags:', '').strip().lower()
                                if search_query in tags:
                                    content = file.read()
                                    # Convert Markdown to HTML
                                    html_content = markdown.markdown(content)
                                    filtered_data.append({'name': filename, 'tags': tags, 'content': html_content})

    return render_template("search.html", filtered_data=filtered_data), 200

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 400)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 400)

        # Query database for username
        rows = db.execute(
            "SELECT * FROM users WHERE username = ?", request.form.get("username").lower()
        )
    
        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(
            rows[0]["hash"], request.form.get("password")
        ):
            return apology("invalid username and/or password", 400)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    elif request.method == "GET":
        return render_template("login.html")
    else:
        return redirect("/")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    session.clear()
    if request.method == "POST":
        if (
            not request.form.get("username")
            or not request.form.get("password")
            or not request.form.get("confirmation")
        ):
            return apology("missing detail", 400)
        if request.form.get("confirmation") != request.form.get("password"):
            return apology("password not matching", 400)
        try:
            user_id = db.execute(
                "INSERT INTO users (username, hash) VALUES(?, ?)",
                request.form.get("username").lower(),
                generate_password_hash(request.form.get("password")),
            )
            session["user_id"] = user_id
            return redirect("/")
        except:
            return apology("user already exists", 400)
    elif request.method == "GET":
        return render_template("register.html")

conversation_history = []
@app.route("/eli5", methods=["POST"])
def eli5():
    global conversation_history

    data = request.json
    selected_text = data.get("text", "")
    system_message = "Use a formal tone and do not introduce yourself. You are a PhD Student in Deep Learning. Your explanations can contain technical jargon to make the concepts clear."
    
    if selected_text:
        # Initialize conversation history if it's the start of a new session
        if not conversation_history:
            conversation_history.append({"role": "system", "content": system_message})
        
        # Add the user's message to the history
        conversation_history.append({"role": "user", "content": selected_text})

        try:
            # Get the response from the model
            response = ollama.chat(
                model="llama3.1",
                messages=conversation_history
            )
            
            # Extract the response content
            explanation = response.get('message', {}).get('content', '').strip()
            
            # Add the assistant's response to the conversation history
            conversation_history.append({"role": "assistant", "content": explanation})
            
            return jsonify({"explanation": explanation})
        
        except Exception as e:
            # Handle any errors from the model or request
            return jsonify({"error": str(e)}), 500

    return jsonify({"explanation": "No text provided"}), 400

@app.route('/save_markdown', methods=['POST'])
def save_markdown():
    data = request.json
    content = data.get('content')
    tags = data.get('tags', '')  # Default to empty string if no tags are provided

    if content:
        # Generate a unique filename with a UUID
        filename = f"{uuid.uuid4()}.md"
        filepath = os.path.join(TAGS_FOLDER, filename)

        try:
            # Ensure the directory exists
            os.makedirs(TAGS_FOLDER, exist_ok=True)

            # Save the content as a markdown file
            with open(filepath, 'w') as file:
                if tags:
                    file.write(f"Tags: {tags}\n\n")
                file.write(content)

            return jsonify({"status": "success", "message": f"File saved as {filename}"}), 200
        
        except Exception as e:
            # Handle any errors that occur during file writing
            return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500
    
    else:
        return jsonify({"status": "error", "message": "No content provided"}), 400

@app.route('/filter_by_tag', methods=['GET'])
def filter_by_tag():
    query_tag = request.args.get('tag')
    matching_files = []

    for filename in os.listdir(TAGS_FOLDER):
        filepath = os.path.join(TAGS_FOLDER, filename)
        with open(filepath, 'r') as file:
            content = file.read()
            if f"Tags: {query_tag}" in content:
                matching_files.append(filename)

    return jsonify({"files": matching_files}), 200

@app.route("/upload_pdf", methods=["GET", "POST"])
@login_required
def upload_pdf():
    if request.method == "POST":
        if 'pdfFile' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'}), 400

        file = request.files['pdfFile']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400

        if file and file.filename.endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)  # Save the file to the server

            # Return the file's URL for rendering
            return jsonify({'success': True, 'message': 'Upload successful!', 'file_url': f'/uploads/{file.filename}'})
        else:
            return jsonify({'success': False, 'message': 'Invalid file type'}), 400
    elif request.method == "GET":
        return render_template("upload.html")
    
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)