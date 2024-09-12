import os
import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    jsonify,
    send_from_directory,
    url_for,
)
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from helpers import (
    apology,
    login_required,
    lookup_titles,
    extract_text_from_pdf,
    remove_section_from_text,
    remove_references_from_text,
    db,
)
import ollama
import uuid
import markdown
import networkx as nx
import faiss
import numpy as np
import igraph as ig
from sentence_transformers import SentenceTransformer
import leidenalg

# Use a more suitable model for scientific papers
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Configure application
app = Flask(__name__)



TAGS_FOLDER = "tags"
os.makedirs(TAGS_FOLDER, exist_ok=True)
# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
UPLOAD_FOLDER = "upload_files"  # Directory to save files
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
vector_store = None
Session(app)

# Use a more suitable model for scientific papers
def build_knowledge_graph(chunks):
    G = nx.Graph()
    model = SentenceTransformer(MODEL_NAME)

    for i, chunk in enumerate(chunks):
        G.add_node(i, content=chunk)

        # Create edges based on semantic similarity
        if i > 0:
            for j in range(i):
                embeddings = model.encode([chunks[i], chunks[j]])
                similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
                if similarity > 0.5:  # Threshold for edge creation
                    G.add_edge(i, j, weight=similarity)

    return G

def hierarchical_clustering(graph):
    ig_graph = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    communities = [frozenset(community) for community in partition]
    return communities

def summarize_communities(communities, graph):
    summaries = {}
    for i, community in enumerate(communities):
        subgraph = graph.subgraph(community)
        summary = generate_community_summary(subgraph)
        summaries[f"community_{i}"] = summary
    return summaries

def generate_community_summary(subgraph):
    content = " ".join([subgraph.nodes[node]['content'] for node in subgraph.nodes])
    prompt = f"Summarize the following content in 2-3 sentences:\n\n{content}"
    response = ollama.generate(model="llama3.1", prompt=prompt)
    return response['response']

def create_embeddings(chunks):
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(chunks), model

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def process_query(query, graph, summaries, index, chunks, embedding_model):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, k=5)
    relevant_chunks = [chunks[i] for i in indices[0]]
    
    # Use relevant chunks, graph information, and community summaries to generate a response
    response = generate_response_with_ollama(query, relevant_chunks, graph, summaries)
    return response

def generate_response_with_ollama(query, relevant_chunks, graph, summaries):
    context = "\n".join(relevant_chunks)
    graph_info = f"Graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges"
    summary_info = "\n".join([f"{k}: {v}" for k, v in summaries.items()])

    prompt = f"""
    Query: {query}
    Context: {context}
    Graph Information: {graph_info}
    Community Summaries: {summary_info}

    Based on the above information, provide a comprehensive answer to the query.
    """
    
    response = ollama.generate(model="llama3.1", prompt=prompt)
    return response['response']

def initialize_vector_store(pdf_filename):
    global vector_store

    file_path = os.path.join(UPLOAD_FOLDER, pdf_filename)

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Extract text from the uploaded PDF
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    text = ""

    # Extract content from the first four pages
    for i, page in enumerate(data):
        text += page.page_content

    # Remove the "Literature Review" section
    cleaned_text = remove_section_from_text(text, "Literature Review")

    # Remove the references and in-text citations
    cleaned_text = remove_references_from_text(cleaned_text)

    if not cleaned_text:
        raise ValueError("No text extracted from the PDF.")

    chunks = cleaned_text.split("\n\n")  # Split cleaned text into chunks
    vector_store = {"chunks": chunks}

    # Build knowledge graph from chunks
    graph = build_knowledge_graph(chunks)
    communities = hierarchical_clustering(graph)
    summaries = summarize_communities(communities, graph)

    # Create embeddings for chunks and build FAISS index
    embeddings, embedding_model = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    # Store all necessary components in the vector store
    vector_store.update({
        "graph": graph,
        "summaries": summaries,
        "index": index,
        "embedding_model": embedding_model
    })




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
    TAGS_FOLDER = "tags"
    saved_data = []

    if os.path.exists(TAGS_FOLDER):
        for filename in os.listdir(TAGS_FOLDER):
            if filename.endswith(".md"):
                file_path = os.path.join(TAGS_FOLDER, filename)
                with open(file_path, "r") as file:
                    # Extract the tags from the markdown file (assuming the first line is the tags)
                    first_line = file.readline().strip()
                    if first_line.startswith("Tags:"):
                        tags = first_line.replace("Tags:", "").strip()
                    else:
                        tags = "No tags"

                    # Read the rest of the file content
                    content = file.read()

                # Convert Markdown to HTML
                html_content = markdown.markdown(content)

                saved_data.append(
                    {"name": filename, "tags": tags, "content": html_content}
                )

    return render_template("history.html", saved_data=saved_data), 200


@app.route("/search", methods=["GET", "POST"])
@login_required
def search():
    """Search markdown files by tags"""
    TAGS_FOLDER = "tags"
    filtered_data = []

    if request.method == "POST":
        search_query = request.form.get("search_query", "").strip().lower()

        if search_query:
            if os.path.exists(TAGS_FOLDER):
                for filename in os.listdir(TAGS_FOLDER):
                    if filename.endswith(".md"):
                        file_path = os.path.join(TAGS_FOLDER, filename)
                        with open(file_path, "r") as file:
                            first_line = file.readline().strip()
                            if first_line.startswith("Tags:"):
                                tags = first_line.replace("Tags:", "").strip().lower()
                                if search_query in tags:
                                    content = file.read()
                                    # Convert Markdown to HTML
                                    html_content = markdown.markdown(content)
                                    filtered_data.append(
                                        {
                                            "name": filename,
                                            "tags": tags,
                                            "content": html_content,
                                        }
                                    )

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
            "SELECT * FROM users WHERE username = ?",
            request.form.get("username").lower(),
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
            response = ollama.chat(model="llama3.1", messages=conversation_history)

            # Extract the response content
            explanation = response.get("message", {}).get("content", "").strip()

            # Add the assistant's response to the conversation history
            conversation_history.append({"role": "assistant", "content": explanation})

            return jsonify({"explanation": explanation})

        except Exception as e:
            # Handle any errors from the model or request
            return jsonify({"error": str(e)}), 500

    return jsonify({"explanation": "No text provided"}), 400


@app.route("/save_markdown", methods=["POST"])
def save_markdown():
    data = request.json
    content = data.get("content")
    tags = data.get("tags", "")  # Default to empty string if no tags are provided

    if content:
        # Generate a unique filename with a UUID
        filename = f"{uuid.uuid4()}.md"
        filepath = os.path.join(TAGS_FOLDER, filename)

        try:
            # Ensure the directory exists
            os.makedirs(TAGS_FOLDER, exist_ok=True)

            # Save the content as a markdown file
            with open(filepath, "w") as file:
                # if tags:
                #     file.write(f"Tags: {tags}\n\n")
                file.write(content)

            return (
                jsonify({"status": "success", "message": f"File saved as {filename}"}),
                200,
            )

        except Exception as e:
            # Handle any errors that occur during file writing
            return (
                jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}),
                500,
            )

    else:
        return jsonify({"status": "error", "message": "No content provided"}), 400


@app.route("/filter_by_tag", methods=["GET"])
def filter_by_tag():
    query_tag = request.args.get("tag")
    matching_files = []

    for filename in os.listdir(TAGS_FOLDER):
        filepath = os.path.join(TAGS_FOLDER, filename)
        with open(filepath, "r") as file:
            content = file.read()
            if f"Tags: {query_tag}" in content:
                matching_files.append(filename)

    return jsonify({"files": matching_files}), 200


@app.route("/write_notes", methods=["GET", "POST"])
@login_required
def write_notes():
    if request.method == "POST":
        content = request.form.get("content")
        tags = request.form.get(
            "tags", ""
        )  # Default to empty string if no tags are provided

        if content:
            # Generate a unique filename with a UUID
            filename = f"{uuid.uuid4()}.md"
            filepath = os.path.join(TAGS_FOLDER, filename)

            try:
                # Ensure the directory exists
                os.makedirs(TAGS_FOLDER, exist_ok=True)

                # Save the content as a markdown file
                with open(filepath, "w") as file:
                    if tags:
                        file.write(f"Tags: {tags}\n\n")
                    file.write(content)

                flash(f"File saved as {filename}", "success")
                return redirect("/history")

            except Exception as e:
                # Handle any errors that occur during file writing
                flash(f"An error occurred: {str(e)}", "danger")
                return redirect("/write_notes")

        else:
            flash("No content provided", "warning")
            return redirect("/write_notes")

    elif request.method == "GET":
        return render_template("write_notes.html", text=None)


@app.route("/upload_pdf", methods=["GET", "POST"])
@login_required
def upload_pdf():
    if request.method == "POST":
        if "pdfFile" not in request.files:
            return jsonify({"success": False, "message": "No file part"}), 400

        file = request.files["pdfFile"]

        if file.filename == "":
            return jsonify({"success": False, "message": "No selected file"}), 400

        if file and file.filename.endswith(".pdf"):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)  # Save the file to the server

            # Extract text from the PDF
            text = extract_text_from_pdf(file_path)

            # Initialize vector store with the new PDF
            try:
                initialize_vector_store(file.filename)
                return jsonify(
                    {
                        "success": True,
                        "message": "Upload successful!",
                        "file_url": f"/uploads/{file.filename}",
                        "extracted_text": text,
                    }
                )
            except Exception as e:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Error initializing vector store: {str(e)}",
                        }
                    ),
                    500,
                )
        else:
            return jsonify({"success": False, "message": "Invalid file type"}), 400
    elif request.method == "GET":
        return render_template("upload.html", text=None)


@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/edit_note/<filename>", methods=["GET", "POST"])
@login_required
def edit_note(filename):
    file_path = os.path.join(TAGS_FOLDER, filename)

    if request.method == "POST":
        content = request.form.get("content")
        tags = request.form.get("tags", "")

        if content:
            try:
                with open(file_path, "w") as file:
                    if tags:
                        file.write(f"Tags: {tags}\n\n")
                    file.write(content)
                flash("Note updated successfully!", "success")
                return redirect("/history")
            except Exception as e:
                flash(f"Error updating note: {str(e)}", "danger")
                return redirect(f"/edit_note/{filename}")
        else:
            flash("No content provided", "warning")
            return redirect(f"/edit_note/{filename}")

    elif request.method == "GET":
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                lines = file.readlines()
                # Extract tags from the first line
                tags_line = lines[0].strip()
                if tags_line.startswith("Tags:"):
                    tags = tags_line.replace("Tags:", "").strip()
                else:
                    tags = ""

                # Extract content (excluding tags line)
                content = "".join(lines[1:])  # Join remaining lines as content

            return render_template(
                "edit_note.html", filename=filename, content=content, tags=tags
            )
        else:
            flash("File not found", "danger")
            return redirect("/history")


@app.route("/delete_note/<filename>", methods=["POST"])
@login_required
def delete_note(filename):
    file_path = os.path.join(TAGS_FOLDER, filename)

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            flash("Note deleted successfully!", "success")
        else:
            flash("File not found", "danger")
    except Exception as e:
        flash(f"Error deleting note: {str(e)}", "danger")

    return redirect("/history")


@app.route("/rag_query", methods=["POST"])
@login_required
def rag():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    if vector_store is None:
        return jsonify({"error": "Vector store is not initialized"}), 500

    # Retrieve components from the vector store
    graph = vector_store["graph"]
    summaries = vector_store["summaries"]
    index = vector_store["index"]
    embedding_model = vector_store["embedding_model"]
    chunks = vector_store["chunks"]

    try:
        # Process the query using the GraphRAG-inspired method
        response = process_query(query, graph, summaries, index, chunks, embedding_model)
        
        return jsonify({"answer": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pin", methods=["POST"])
@login_required
def pin_paper():
    paper_id = request.form.get("paper_id")
    published = request.form.get("published")
    submitted_by = request.form.get("submitted_by")
    summary = request.form.get("summary")
    upvotes = request.form.get("upvotes")

    # Check if the paper is already pinned for the user
    pinned = db.execute(
        "SELECT * FROM pinned_papers WHERE paper_id = ? AND user_id = ?",
        paper_id,
        session["user_id"],
    )

    if pinned:
        # If the paper is already in the table, update its pinned status
        db.execute(
            "UPDATE pinned_papers SET pinned = 1 WHERE paper_id = ? AND user_id = ?",
            paper_id,
            session["user_id"],
        )
        flash(f"{paper_id} has been pinned.")
    else:
        # Insert a new record if it doesn't exist
        db.execute(
            "INSERT INTO pinned_papers (user_id, paper_id, published, submitted_by, summary, upvotes, pinned) VALUES (?, ?, ?, ?, ?, ?, 1)",
            session["user_id"],
            paper_id,
            published,
            submitted_by,
            summary,
            upvotes,
        )
        flash(f"{paper_id} has been pinned.")

    return redirect(url_for("index"))


@app.route("/unpin/<string:paper_id>", methods=["POST"])
@login_required
def unpin_paper(paper_id):
    # Check if the paper exists in the pinned_papers table
    paper = db.execute(
        "SELECT paper_id FROM pinned_papers WHERE paper_id = ? AND user_id = ?",
        paper_id,
        session["user_id"],
    )

    if paper:
        # Update the paper to be unpinned
        db.execute(
            "UPDATE pinned_papers SET pinned = 0 WHERE paper_id = ? AND user_id = ?",
            paper_id,
            session["user_id"],
        )
        flash(f"{paper_id} has been unpinned.")
    else:
        flash("Paper not found.")

    return redirect(url_for("index"))
