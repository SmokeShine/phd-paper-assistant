import csv
import datetime
import numpy as np
import pytz
import requests
import os
import pickle
from sentence_transformers import SentenceTransformer
import feedparser
from bs4 import BeautifulSoup, Comment
import faiss
import urllib
import re
import uuid
import json
from pandas import json_normalize
import pandas as pd
from flask import redirect,flash, render_template, request, session
from functools import wraps
import fitz
from langchain_community.document_loaders import PyMuPDFLoader
import ollama
from cs50 import SQL
import diskcache as dc  # Adding diskcache for caching
from contextlib import contextmanager
import sqlite3
from flask_apscheduler import APScheduler

CACHE_DIR = "conference_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")
scheduler = APScheduler()

def load_from_cache(file_path):
    """Load data from cache if it exists."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None

def save_to_cache(file_path, data):
    """Save data to cache (disk)."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def fetch_json_from_url(url, cache_file_path):
    """Fetch JSON data from URL or cache."""
    cached_data = load_from_cache(cache_file_path)
    if cached_data is not None:
        return cached_data
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        save_to_cache(cache_file_path, data)  # Save fetched data to disk cache
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None
def load_conference_papers(conference_name):
    """
    Loads papers from a conference's GitHub repository, filters by accepted status,
    and sorts them by year (latest first).
    """
    api_url = f"https://api.github.com/repos/papercopilot/paperlists/contents/{conference_name}"
    raw_url = f"https://raw.githubusercontent.com/papercopilot/paperlists/main/{conference_name}/"

    try:
        # Cache the file list (list of .json files)
        file_list_cache_path = os.path.join(CACHE_DIR, f"{conference_name}_file_list.pkl")
        file_list = fetch_json_from_url(api_url, file_list_cache_path)
        if not file_list:
            return []

        files = [file["name"] for file in file_list if file["name"].endswith(".json")]

        papers = []
        for file in files:
            file_url = f"{raw_url}{file}"
            file_cache_path = os.path.join(CACHE_DIR, f"{conference_name}_{file}.pkl")
            data = fetch_json_from_url(file_url, file_cache_path)

            if not data:
                continue  # Skip if failed to fetch

            # Extract year from filename (assuming format "confYEAR.json")
            year_str = "".join(filter(str.isdigit, file))
            year = int(year_str) if year_str.isdigit() else None
            if year is None or year <= 2022:
                continue

            # Filter out papers with "Reject" or "Withdraw" status
            accepted_papers = [
                {**paper, "year": year}
                for paper in data
                if paper.get("status") not in {"Reject", "Withdraw"}
            ]
            papers.extend(accepted_papers)

        # Sort papers by year (latest first)
        return sorted(papers, key=lambda x: x["year"], reverse=True)

    except Exception as e:
        print(f"Error loading conference papers: {e}")
        return []

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {conference_name} papers: {e}")
        return []





# Initialize disk cache
cache = dc.Cache("cache_directory")  # Specify a directory for cache storage

CACHE_EXPIRATION_DAYS = 1  # Cache expiration period
CACHE_KEY_DATA = "lookup_titles_data"
CACHE_KEY_TIMESTAMP = "lookup_titles_timestamp"


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

    # Query the database for pinned papers (should be done regardless of cache validity)
    pinned_papers_db = pd.DataFrame(
        db.execute(
            "SELECT id, paper_id as title, published, submitted_by, summary, upvotes FROM pinned_papers WHERE pinned = 1 AND user_id = ?",
            session["user_id"],
        )
    )
    
    # Check if the query returned any results
    if not pinned_papers_db.empty:
        # If there are rows, add the 'pinned' column
        pinned_papers_db["pinned"] = 1
    else:
        # If no pinned papers, ensure the DataFrame has the expected structure with empty columns
        pinned_papers_db = pd.DataFrame(columns=["id", "title", "published", "submitted_by", "summary", "upvotes", "pinned"])

    # Check if the cache data exists and is still valid
    if CACHE_KEY_TIMESTAMP in cache:
        cache_time = cache[CACHE_KEY_TIMESTAMP]

        if (current_time - cache_time).days <= CACHE_EXPIRATION_DAYS:
            # Get the cached data
            cached_papers = pd.DataFrame(cache[CACHE_KEY_DATA])

            # Check for any inconsistencies between cached pinned papers and DB pinned papers
            cached_pinned = cached_papers[cached_papers["pinned"] == 1]

            # Condition 1: Update cache if a paper is pinned in DB but not in the cache
            db_not_in_cache = pinned_papers_db[~pinned_papers_db["title"].isin(cached_pinned["title"])]
            
            if not db_not_in_cache.empty:
                # Update cached papers with pinned status
                for index, row in db_not_in_cache.iterrows():
                    new_paper = row.to_dict()
                    new_paper["pinned"] = 1
                    cached_papers = pd.concat([cached_papers,pd.DataFrame([new_paper])], ignore_index=True)

            # Condition 2: Update cache if a paper is pinned in cache but not in DB
            cache_not_in_db = cached_pinned[~cached_pinned["title"].isin(pinned_papers_db["title"])]
            if not cache_not_in_db.empty:
                # Set pinned status to 0 for papers no longer pinned in the DB
                for index, row in cache_not_in_db.iterrows():
                    cached_papers.loc[cached_papers['title'] == row['title'], 'pinned'] = 0

            # Now sort the cached papers: pinned first, then unpinned by upvotes
            cached_papers.sort_values(by=["pinned", "upvotes"], ascending=[False, False], inplace=True)

            # Convert the updated DataFrame back to a dictionary format
            papers = cached_papers.to_dict(orient="records")

            # Update the cache with the modified data
            cache[CACHE_KEY_DATA] = papers
            cache[CACHE_KEY_TIMESTAMP] = current_time

            return papers

    # If cache is expired or doesn't exist, fetch new data
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

        papers_data["pinned"] = 0  # Initialize all as unpinned

        # If there are pinned papers, merge them with the fetched papers
        if not pinned_papers_db.empty:
            papers_data = pd.concat([pinned_papers_db, papers_data])

            # Drop duplicates based on key columns
            papers_data.drop_duplicates(
                ["title", "published", "submitted_by", "summary", "upvotes"],
                inplace=True,
            )

        # Sort: pinned papers on top, then unpinned by upvotes
        papers_data.sort_values(by=["pinned", "upvotes"], ascending=[False, False], inplace=True)

        # Convert the final DataFrame to a dictionary format
        papers = papers_data.to_dict(orient="records")

        # Update the cache with the new data and timestamp
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
        model="llama3.2",
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

    return cleaned_text


class VectorStore:
    def __init__(self):
        self.store = {}

    def save(self, key, value):
        if key is None or value is None:
            raise ValueError("Key and value must not be None")
        self.store[key] = value

    def load(self, key):
        return self.store.get(key)

    def exists(self, key):
        return key in self.store
    
    def clear(self):
        self.store.clear()
    
    def save_to_disk(self, filename):
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self.store, f)

    def load_from_disk(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")
        with open(filename, "rb") as f:
            self.store = pickle.load(f)
        
class DatabaseTransactionManager:
    def __init__(self, db):
        self.db = db
    
    @contextmanager
    def transaction(self):
        """Context manager for handling database transactions"""
        try:
            # Begin transaction
            self.db.execute("BEGIN TRANSACTION")
            yield
            # Commit if no exceptions occurred
            self.db.execute("COMMIT")
        except Exception as e:
            # Rollback on error
            self.db.execute("ROLLBACK")
            raise e

class RSSFeedManager:
    def __init__(self, db, cache_dir="cache",refresh_interval=30):
        """Initialize RSS Feed Manager with database connection and cache directory"""
        self.db = db
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.refresh_interval = refresh_interval
        self.setup_auto_refresh()


    def get_user_feeds(self, user_id):
        """Get all RSS feeds for a user"""
        return self.db.execute("""
            SELECT * FROM rss_feeds 
            WHERE user_id = ? 
            ORDER BY name
        """, user_id)

    def add_feed(self, user_id, name, url, category):
        """Add a new RSS feed and fetch its initial articles"""
        cache_file = os.path.join(self.cache_dir, f"feed_{hash(url)}.pickle")
        
        try:
            # Try to get from cache first
            # import pdb;pdb.set_trace()
            feed_data = self._load_from_cache(cache_file)
            if not feed_data:
                feed = feedparser.parse(url)
                if hasattr(feed, 'bozo') and feed.bozo:
                    return {"success": False, "error": "Invalid feed URL"}
                feed_data = feed
                self._save_to_cache(cache_file, feed)

            # Add feed to database
            feed_id = self.db.execute("""
                INSERT INTO rss_feeds (user_id, name, url, category, last_updated)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, user_id, name, url, category)

            # Get the last inserted ID
            result = self.db.execute("SELECT last_insert_rowid() as id")
            feed_id = result[0]['id']

            # Store the articles
            # import pdb;pdb.set_trace()
            self._store_feed_articles(feed_id, feed_data.entries)
            return {"success": True, "feed_id": feed_id}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_feed_articles(self, user_id, category=None, sort="newest"):
        """Get articles from user's feeds with optional filtering and sorting"""
        query = """
            SELECT 
                a.*, f.name as feed_name, f.category
            FROM rss_articles a
            JOIN rss_feeds f ON a.feed_id = f.id
            WHERE f.user_id = ?
        """
        params = [user_id]
        
        if category and category != "all":
            query += " AND f.category = ?"
            params.append(category)
        
        query += " ORDER BY a.published_date"
        if sort == "newest":
            query += " DESC"
        
        return self.db.execute(query, *params)
    def setup_auto_refresh(self):
        """Setup automatic refresh scheduler"""
        @scheduler.task('interval', id='refresh_feeds', minutes=self.refresh_interval, misfire_grace_time=900)
        def auto_refresh_feeds():
            print(f"Auto-refreshing feeds at {datetime.datetime.now()}")
            stale_feeds = self.db.execute("""
                SELECT * FROM rss_feeds 
                WHERE last_updated IS NULL 
                OR datetime(last_updated) <= datetime('now', '-30 minutes')
            """)
            
            for feed in stale_feeds:
                try:
                    self.refresh_feed(feed['id'])
                except Exception as e:
                    print(f"Error auto-refreshing feed {feed['id']}: {str(e)}")
            
            print(f"Finished auto-refresh at {datetime.datetime.now()}")

    def refresh_feed(self, feed_id):
        """Refresh a specific feed"""
        try:
            feed = self.db.execute("SELECT * FROM rss_feeds WHERE id = ?", feed_id)
            if not feed or len(feed) == 0:
                return {"success": False, "error": "Feed not found"}
            
            feed = feed[0]
            feed_data = feedparser.parse(feed['url'])
            
            if not hasattr(feed_data, 'bozo') or not feed_data.bozo:
                self._store_feed_articles(feed['id'], feed_data.entries)
                
                self.db.execute("""
                    UPDATE rss_feeds 
                    SET last_updated = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, feed['id'])
                
                return {"success": True, "message": f"Feed {feed['name']} refreshed successfully"}
            
            return {"success": False, "error": "Invalid feed data"}
            
        except Exception as e:
            print(f"Error refreshing feed: {str(e)}")
            return {"success": False, "error": str(e)}
    def refresh_all_feeds(self, user_id=None):
        """Refresh all feeds for a user or all feeds if user_id is None"""
        try:
            # Get feeds to refresh
            if user_id:
                feeds = self.db.execute("""
                    SELECT * FROM rss_feeds 
                    WHERE user_id = ?
                """, user_id)
            else:
                feeds = self.db.execute("SELECT * FROM rss_feeds")

            refreshed_count = 0
            for feed in feeds:
                try:
                    # Parse feed
                    feed_data = feedparser.parse(feed['url'])
                    if not hasattr(feed_data, 'bozo') or not feed_data.bozo:
                        # Store new articles
                        self._store_feed_articles(feed['id'], feed_data.entries)
                        
                        # Update last_updated timestamp
                        self.db.execute("""
                            UPDATE rss_feeds 
                            SET last_updated = CURRENT_TIMESTAMP 
                            WHERE id = ?
                        """, feed['id'])
                        
                        refreshed_count += 1
                except Exception as e:
                    print(f"Error refreshing feed {feed['url']}: {str(e)}")
                    continue

            return {
                "success": True, 
                "message": f"Successfully refreshed {refreshed_count} feeds",
                "refreshed_count": refreshed_count
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def refresh_feed(self, feed_id):
        """Refresh a specific feed"""
        try:
            # Get feed info
            feed = self.db.execute("SELECT * FROM rss_feeds WHERE id = ?", feed_id)
            if not feed or len(feed) == 0:
                return {"success": False, "error": "Feed not found"}
            
            feed = feed[0]
            
            # Parse and store new articles
            feed_data = feedparser.parse(feed['url'])
            if not hasattr(feed_data, 'bozo') or not feed_data.bozo:
                self._store_feed_articles(feed['id'], feed_data.entries)
                
                # Update timestamp
                self.db.execute("""
                    UPDATE rss_feeds 
                    SET last_updated = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, feed['id'])
                
                return {"success": True, "message": "Feed refreshed successfully"}
            else:
                return {"success": False, "error": "Invalid feed data"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _store_feed_articles(self, feed_id, entries):
        """Store articles from feed entries"""
        try:
            # Get feed info using CS50 SQL
            feed_info = self.db.execute("SELECT * FROM rss_feeds WHERE id = ?", feed_id)
            if not feed_info or len(feed_info) == 0:
                return False
                
            feed_info = feed_info[0]  # Get the first row since CS50 SQL returns a list
            
            for entry in entries:
                try:
                    # Parse publication date with fallback
                    try:
                        published = entry.get('published', entry.get('updated'))
                        if published:
                            pub_date = datetime.datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %z')
                        else:
                            pub_date = datetime.datetime.now(pytz.UTC)
                    except (ValueError, AttributeError):
                        pub_date = datetime.datetime.now(pytz.UTC)
                    
                    # Check if article already exists
                    existing_article = self.db.execute("""
                        SELECT id FROM rss_articles 
                        WHERE feed_id = ? AND link = ?
                    """, feed_id, entry.get('link', ''))
                    
                    if not existing_article:
                        # Insert only if article doesn't exist
                        self.db.execute("""
                            INSERT INTO rss_articles 
                            (feed_id, title, description, link, published_date)
                            VALUES (?, ?, ?, ?, ?)
                        """, 
                            feed_id,
                            entry.get('title', 'Untitled'),
                            entry.get('description', entry.get('summary', '')),
                            entry.get('link', ''),
                            pub_date.isoformat()
                        )
                    
                except Exception as e:
                    if not "UNIQUE constraint failed" in str(e):
                        print(f"Error storing article: {str(e)}")
                    continue
                    
            return True
        
        except Exception as e:
            print(f"Error in _store_feed_articles: {str(e)}")
            return False

    def _load_from_cache(self, file_path):
        """Load data from cache if it exists and is fresh"""
        try:
            if os.path.exists(file_path):
                if (datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(file_path))).seconds < 3600:
                    with open(file_path, "rb") as f:
                        return pickle.load(f)
        except Exception as e:
            print(f"Cache load error: {str(e)}")
        return None

    def _save_to_cache(self, file_path, data):
        """Save data to cache file"""
        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Cache save error: {str(e)}")
    def delete_feed(self, user_id, feed_id):
        """Delete an RSS feed and its articles"""
        try:
            self.db.execute("DELETE FROM rss_articles WHERE feed_id = ?", feed_id)
            self.db.execute("""
                DELETE FROM rss_feeds 
                WHERE id = ? AND user_id = ?
            """, feed_id, user_id)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}