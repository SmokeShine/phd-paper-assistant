import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from flask_apscheduler import APScheduler
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
    Response,
)
from datetime import datetime
import pytz
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from helpers import (
    apology,
    load_conference_papers,
    login_required,
    lookup_titles,
    extract_text_from_pdf,
    RSSFeedManager,
    db,
    VectorStore,
    scheduler,
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
model = SentenceTransformer(MODEL_NAME)

# Configure application
app = Flask(__name__)
app.config["SCHEDULER_API_ENABLED"] = True
# Initialize scheduler
scheduler.init_app(app)
scheduler.start()
rss_manager = RSSFeedManager(db, refresh_interval=30)

TAGS_FOLDER = "tags"
os.makedirs(TAGS_FOLDER, exist_ok=True)
# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
UPLOAD_FOLDER = "upload_files"  # Directory to save files
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

vector_store = VectorStore()
Session(app)

# Read and execute the SQL file
with open("init_rss_db.sql", "r") as sql_file:
    sql_commands = sql_file.read()
    # Split commands and execute each one
    for command in sql_commands.split(";"):
        if command.strip():
            db.execute(command)


def hierarchical_clustering(graph):
    ig_graph = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    communities = [frozenset(community) for community in partition]
    return communities


def split_documents_into_chunks(documents, chunk_size=600, overlap_size=100):
    chunks = []
    for document in documents:
        for i in range(0, len(document), chunk_size - overlap_size):
            chunk = document[i : i + chunk_size]
            chunks.append(chunk)
    return chunks


def extract_elements_from_chunks(chunks):
    elements = []
    for index, chunk in enumerate(chunks):
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": "Extract entities and relationships from the following text.",
                },
                {"role": "user", "content": chunk},
            ],
        )

        entities_and_relations = response["message"]["content"]
        elements.append(entities_and_relations)
    return elements


def summarize_elements(elements):
    summaries = []
    for index, element in enumerate(elements):
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": 'Summarize the following entities and relationships in a structured format. Use "->" to represent relationships, after the "Relationships:" word.',
                },
                {"role": "user", "content": element},
            ],
        )
        summary = response["message"]["content"]
        summaries.append(summary)
    return summaries


def build_graph_from_summaries(summaries):
    G = nx.Graph()
    for index, summary in enumerate(summaries):
        lines = summary.split("\n")
        entities_section = False
        relationships_section = False
        entities = []
        for line in lines:
            if line.startswith("### Entities:") or line.startswith("**Entities:**"):
                entities_section = True
                relationships_section = False
                continue
            elif line.startswith("### Relationships:") or line.startswith(
                "**Relationships:**"
            ):
                entities_section = False
                relationships_section = True
                continue
            if entities_section and line.strip():
                if line[0].isdigit() and line[1] == ".":
                    line = line.split(".", 1)[1].strip()
                entity = line.strip()
                entity = entity.replace("**", "")
                entities.append(entity)
                G.add_node(entity)
            elif relationships_section and line.strip():
                parts = line.split("->")
                if len(parts) >= 2:
                    source = parts[0].strip()
                    target = parts[-1].strip()
                    relation = " -> ".join(parts[1:-1]).strip()
                    G.add_edge(source, target, label=relation)
    return G


def detect_communities(graph):
    node_to_index = {node: i for i, node in enumerate(graph.nodes())}
    index_to_node = {i: node for node, i in node_to_index.items()}
    ig_graph = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
    communities = []
    for community in partition:
        communities.append([index_to_node[idx] for idx in community])

    return communities


def summarize_communities(communities, graph):
    community_summaries = []

    for index, community in enumerate(communities):
        subgraph = graph.subgraph(set(community))
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges(data=True))
        description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
        relationships = []
        for edge in edges:
            relationships.append(f"{edge[0]} -> {edge[2]['label']} -> {edge[1]}")
        description += ", ".join(relationships)

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following community of entities and relationships.",
                },
                {"role": "user", "content": description},
            ],
        )
        summary = response["message"]["content"]
        community_summaries.append(summary)
    return community_summaries


def generate_answers_from_communities(summaries_embeddings, community_summaries, query):

    query_embedding = model.encode([query])
    # Compute cosine similarities between query and each community summary
    similarities = cosine_similarity(query_embedding, summaries_embeddings)

    # Find the index of the most similar community summary
    closest_index = similarities.argmax()
    closest_summary = community_summaries[closest_index]

    final_response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": "Combine these answers into a final response.",
            },
            {
                "role": "user",
                "content": f"Query: {query} Summary: {closest_summary}",
            },
        ],
    )
    final_answer = final_response["message"]["content"]
    return final_answer


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
    # Get ICLR papers
    iclr_papers = load_conference_papers("iclr")
    neurips_papers = load_conference_papers("nips")
    icml_papers = load_conference_papers("icml")
    cvpr_papers = load_conference_papers("cvpr")
    # Load RSS feeds and articles
    try:
        # import pdb;pdb.set_trace()
        # Get user's subscribed feeds
        subscribed_feeds = rss_manager.get_user_feeds(session["user_id"])

        # Refresh all feeds to get latest articles
        rss_manager.refresh_all_feeds(session["user_id"])

        # Get all articles from subscribed feeds
        feed_articles = rss_manager.get_feed_articles(
            session["user_id"],
            category=request.args.get("category"),
            sort=request.args.get("sort", "newest"),
        )

    except Exception as e:
        print(f"Error loading RSS content: {str(e)}")
        subscribed_feeds = []
        feed_articles = []

    return (
        render_template(
            "index.html",
            papers_data=papers_data,
            iclr_papers=iclr_papers,
            neurips_papers=neurips_papers,
            icml_papers=icml_papers,
            cvpr_papers=cvpr_papers,
            subscribed_feeds=subscribed_feeds,
            feed_articles=feed_articles,
            is_hugging_face=True,
        ),
        200,
    )


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


@app.route("/eli5", methods=["POST"])
def eli5():
    conversation_history = []
    data = request.json
    selected_text = data.get("text", "")
    system_message = "Use a formal tone and do not introduce yourself. You are a PhD Student in Deep Learning. Your explanations can contain technical jargon to make the concepts clear."

    if selected_text:
        if not conversation_history:
            conversation_history.append({"role": "system", "content": system_message})

        conversation_history.append({"role": "user", "content": selected_text})

        def generate_response():
            try:
                response_stream = ollama.chat(
                    model="llama3.2", messages=conversation_history, stream=True
                )
                for chunk in response_stream:
                    content = chunk.get("message", {}).get("content", "").strip()
                    if content:
                        yield content + "\n"

            except Exception as e:
                yield f"Error: {str(e)}\n"

        return Response(generate_response(), content_type="text/event-stream")

    return jsonify({"error": "No text provided"}), 400


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
        global vector_store
        if "pdfFile" not in request.files:
            return jsonify({"success": False, "message": "No file part"}), 400

        file = request.files["pdfFile"]

        if file.filename == "":
            return jsonify({"success": False, "message": "No selected file"}), 400

        if file and file.filename.endswith(".pdf"):
            # Generate paths for both the PDF file and its corresponding .pkl file
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            vector_store_filename = os.path.splitext(file.filename)[0] + ".pkl"
            vector_store_path = os.path.join(
                app.config["UPLOAD_FOLDER"], vector_store_filename
            )

            # Check if the .pkl file already exists
            if os.path.exists(vector_store_path):
                try:
                    # Load the vector store from disk if the file exists
                    with open(vector_store_path, "rb") as f:
                        vector_store = pickle.load(f)
                        text = extract_text_from_pdf(file_path)
                        session["documents"] = [text]  # Store text in session
                    return jsonify(
                        {
                            "success": True,
                            "message": "Vector store loaded from previous upload!",
                            "file_url": f"/uploads/{file.filename}",
                            "extracted_text": text,
                            "vector_store_file": vector_store_filename,  # Provide the saved vector store filename
                        }
                    )
                except Exception as e:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": f"Error loading vector store: {str(e)}",
                            }
                        ),
                        500,
                    )

            # If the .pkl file doesn't exist, process the PDF and save the vector store
            file.save(file_path)  # Save the file to the server

            # Extract text from the PDF
            text = extract_text_from_pdf(file_path)
            session["documents"] = [text]  # Store text in session

            vector_store.clear()  # Clear the vector store for the new PDF

            try:
                # Save documents to vector store
                vector_store.save("documents", text)

                # Pre-compute and save chunks, elements, summaries, etc.
                chunks = split_documents_into_chunks([text])
                vector_store.save("chunks", chunks)

                elements = extract_elements_from_chunks(chunks)
                vector_store.save("elements", elements)

                summaries = summarize_elements(elements)
                vector_store.save("summaries", summaries)

                graph = build_graph_from_summaries(summaries)
                vector_store.save("graph", graph)

                communities = detect_communities(graph)
                vector_store.save("communities", communities)

                community_summaries = summarize_communities(communities, graph)
                vector_store.save("community_summaries", community_summaries)

                summaries_embeddings = model.encode(community_summaries)
                vector_store.save("summaries_embeddings", summaries_embeddings)

                # Save vector store to disk in Pickle format
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vector_store, f)

                return jsonify(
                    {
                        "success": True,
                        "message": "Upload successful!",
                        "file_url": f"/uploads/{file.filename}",
                        "extracted_text": text,
                        "vector_store_file": vector_store_filename,  # Provide the saved vector store filename
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
def rag_query():
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    try:
        # Load intermediate data from vector store with fallback logic

        documents = vector_store.load("documents")
        if not documents:
            return jsonify({"error": "No documents available for processing"}), 500

        chunks = vector_store.load("chunks") or split_documents_into_chunks([documents])
        elements = vector_store.load("elements") or extract_elements_from_chunks(chunks)
        summaries = vector_store.load("summaries") or summarize_elements(elements)
        graph = vector_store.load("graph") or build_graph_from_summaries(summaries)
        communities = vector_store.load("communities") or detect_communities(graph)
        community_summaries = vector_store.load(
            "community_summaries"
        ) or summarize_communities(communities, graph)
        summaries_embeddings = vector_store.load("summaries_embeddings")
        if summaries_embeddings is None:
            summaries_embeddings = model.encode(community_summaries)
        response = generate_answers_from_communities(
            summaries_embeddings, community_summaries, query
        )

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


@app.route("/delete_feed/<int:feed_id>", methods=["POST"])
@login_required
def delete_feed_route(feed_id):
    result = rss_manager.delete_feed(session["user_id"], feed_id)
    if result["success"]:
        flash("Feed deleted successfully!", "success")
    else:
        flash(f"Error deleting feed: {result['error']}", "error")

    return redirect(url_for("index", _anchor="rss-tab"))


@app.route("/add_rss_feed_route", methods=["POST"])
@login_required
def add_rss_feed_route():
    result = rss_manager.add_feed(
        session["user_id"],
        request.form.get("feed_name"),
        request.form.get("rss_url"),
        request.form.get("category"),
    )
    if result["success"]:
        flash("RSS feed added successfully!", "success")
    else:
        flash(f"Error adding RSS feed: {result['error']}", "error")
    return redirect(url_for("index", _anchor="rss-tab"))


@app.route("/feeds/refresh", methods=["POST"])
@login_required
def refresh_feeds_route():
    import pdb;pdb.set_trace()
    return rss_manager.refresh_all_feeds(session["user_id"])

@app.template_filter('timeago')
def timeago_filter(timestamp):
    """Convert timestamp to "time ago" text"""
    if not timestamp:
        return ''
    
    try:
        # Convert string to datetime if needed
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
        now = datetime.datetime.now(pytz.UTC)
        diff = now - timestamp

        seconds = diff.total_seconds()
        if seconds < 60:
            return 'just now'
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f'{minutes}m ago'
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f'{hours}h ago'
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f'{days}d ago'
        else:
            return timestamp.strftime('%Y-%m-%d')
    except Exception:
        return str(timestamp)