import csv
import datetime
import numpy as np
import pytz
import requests
from sentence_transformers import SentenceTransformer
import faiss
import urllib
import re
import uuid
import json
from pandas import json_normalize
import pandas as pd
from flask import redirect, render_template, request, session
from functools import wraps
import fitz
from langchain_community.document_loaders import PyMuPDFLoader
import ollama
from cs50 import SQL
import diskcache as dc  # Adding diskcache for caching

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")

# Initialize disk cache
cache = dc.Cache('cache_directory')  # Specify a directory for cache storage

CACHE_EXPIRATION_DAYS = 7  # Cache expiration period
CACHE_KEY_DATA = 'lookup_titles_data'
CACHE_KEY_TIMESTAMP = 'lookup_titles_timestamp'

def apology(message, code=400):
    """Render message as an apology to user."""
    def escape(s):
        """
        Escape special characters.
        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [
            ("-", "--"),
            (" ", "-"),
            ("_", "__"),
            ("?", "~q"),
            ("%", "~p"),
            ("#", "~h"),
            ("/", "~s"),
            ('"', "''"),
        ]:
            s = s.replace(old, new)
        return s

    return render_template("apology.html", top=code, bottom=escape(message)), code

def login_required(f):
    """
    Decorate routes to require login.
    https://flask.palletsprojects.com/en/latest/patterns/viewdecorators/
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)

    return decorated_function

def lookup_titles():
    """Look up recent papers from Hugging Face and merge with pinning data from SQLite."""

    current_time = datetime.datetime.now(pytz.timezone("US/Eastern"))

    # Check if the cache data exists and is still valid
    if CACHE_KEY_TIMESTAMP in cache:
        cache_time = cache[CACHE_KEY_TIMESTAMP]
        if (current_time - cache_time).days <= CACHE_EXPIRATION_DAYS:
            return cache[CACHE_KEY_DATA]

    end = current_time
    start = (end - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    url = f"https://huggingface.co/api/daily_papers?date>{start}"

    try:
        response = requests.get(
            url,
            cookies={"session": str(uuid.uuid4())},
            headers={"Accept": "*/*", "User-Agent": request.headers.get("User-Agent")},
        )
        response.raise_for_status()
        quotes = json.loads(response.content.decode("utf-8"))
        flatten_data = pd.json_normalize(quotes)

        title = flatten_data.title
        published_at = pd.to_datetime(flatten_data.publishedAt).dt.date
        submitted_by = flatten_data["submittedBy.fullname"]
        summary = flatten_data["paper.summary"]
        upvotes = flatten_data["paper.upvotes"]
        paper_ids = flatten_data["paper.id"]

        papers_data = pd.concat(
            [paper_ids, title, published_at, submitted_by, summary, upvotes], axis=1
        )
        papers_data.columns = [
            "id",
            "title",
            "published",
            "submitted_by",
            "summary",
            "upvotes",
        ]
        papers_data["summary"] = papers_data["summary"].str.replace("\n", "<br>")
        papers_data.sort_values(by="upvotes", ascending=False, inplace=True)

        papers_data["pinned"] = 0

        # Query the database to get pinned papers
        pinned_papers = pd.DataFrame(db.execute(
                "SELECT id, paper_id as title, published, submitted_by, summary, upvotes FROM pinned_papers WHERE pinned = 1 AND user_id = ?",
                session["user_id"],
            ))
        
        if len(pinned_papers) > 0:
            # Add a new column 'pinned' with value 1
            pinned_papers["pinned"] = 1

            # Assuming 'papers' is also a DataFrame
            papers_data = pd.concat([pinned_papers, papers_data])

            # Drop duplicates based on the specified columns
            papers_data.drop_duplicates(
                ["title", "published", "submitted_by", "summary", "upvotes"],
                inplace=True,
            )

        papers = papers_data.to_dict(orient="records")

        # Update the cache with new data and timestamp
        cache[CACHE_KEY_DATA] = papers
        cache[CACHE_KEY_TIMESTAMP] = current_time
        return papers

    except (KeyError, IndexError, requests.RequestException, ValueError):
        return None

# Function to remove references and citations
def remove_references_from_text(text):
    references_patterns = ["references", "bibliography", "works cited"]
    lower_text = text.lower()
    for pattern in references_patterns:
        ref_index = lower_text.find(pattern)
        if ref_index != -1:
            text = text[:ref_index].strip()
            break

    text = re.sub(r"\[\d+\]", "", text)  # e.g., [1], [12], etc.
    text = re.sub(r"\(\w+ et al\., \d{4}\)", "", text)  # e.g., (Smith et al., 2020)

    return text

# Function to remove specific sections like "Literature Review"
def remove_section_from_text(text, section_heading):
    lower_text = text.lower()
    start_index = lower_text.find(section_heading.lower())

    if start_index == -1:
        return text

    next_section_pattern = (
        r"\n\s*(introduction|methods|results|discussion|conclusion|references)\b"
    )
    match = re.search(next_section_pattern, lower_text[start_index:])

    if match:
        end_index = start_index + match.start()
    else:
        end_index = len(text)

    return text[:start_index].strip() + "\n" + text[end_index:].strip()

# Function to chunk text based on sections
def chunk_text_by_sections(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks

# Extract and preprocess text from PDF
def extract_and_preprocess_text(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    full_text = ""

    # Combine content from all pages
    for page in data:
        full_text += page.page_content

    # Remove unwanted sections and references
    text_without_lit_review = remove_section_from_text(full_text, "Literature Review")
    cleaned_text = remove_references_from_text(text_without_lit_review)

    # Chunk text by sections or by fixed size
    chunks = chunk_text_by_sections(cleaned_text)
    return chunks

# Create embeddings for text chunks
def create_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings, model

# Build a FAISS index
def build_faiss_index(embeddings):
    embedding_matrix = np.array(embeddings)
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    return index

# Retrieve relevant chunks using FAISS
def retrieve_relevant_chunks(query, index, embedding_model, chunks, top_k=5):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = [chunks[idx] for idx in indices[0]]
    return results

# Generate a response using Ollama with the relevant chunks
def generate_response_with_ollama(relevant_chunks, query):
    conversation_history = [
        {
            "role": "system",
            "content": "Use a formal tone and do not introduce yourself. Don't ask any questions at the end. You are a PhD Student in Deep Learning. You need to explain the novelty in this paper.",
        },
        {"role": "user", "content": " ".join(relevant_chunks)},
    ]

    response = ollama.chat(
        model="llama3.1",
        messages=conversation_history,
        options=ollama.Options(context_length=8096),
    )

    summary = response.get("message", {}).get("content", "").strip()
    return summary

# Main function to process the PDF and generate a summary or answer
def process_ml_paper(pdf_path, query):
    # Step 1: Extract and preprocess text
    chunks = extract_and_preprocess_text(pdf_path)

    # Step 2: Create embeddings
    embeddings, embedding_model = create_embeddings(chunks)

    # Step 3: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 4: Retrieve relevant chunks based on the query
    relevant_chunks = retrieve_relevant_chunks(query, index, embedding_model, chunks)

    # Step 5: Generate a response using Ollama
    summary = generate_response_with_ollama(relevant_chunks, query)

    return summary

def extract_text_from_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    text = ""

    # Extract content from the first four pages
    for i, page in enumerate(data):
        text += page.page_content

    # Remove the "Literature Review" section
    cleaned_text = remove_section_from_text(text, "Literature Review")

    # Remove the references and in-text citations
    cleaned_text = remove_references_from_text(cleaned_text)

    # Prepare conversation history for Ollama
    conversation_history = [
        {
            "role": "system",
            "content": "Use a formal tone and do not introduce yourself. Don't ask any questions at the end. You are a PhD Student in Deep Learning. You need to explain the novelty in this paper.",
        },
        {"role": "user", "content": cleaned_text},
    ]

    # Generate a response using Ollama
    response = ollama.chat(
        model="llama3.1",
        messages=conversation_history,
        options=ollama.Options(context_length=8096),
    )

    # Extract and return the response content
    summary = response.get("message", {}).get("content", "").strip()
    return summary
