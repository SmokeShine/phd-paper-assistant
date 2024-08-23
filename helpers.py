import csv
import datetime
import pytz
import requests
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
    """Look up quote for symbol."""

    # Prepare API request
    end = datetime.datetime.now(pytz.timezone("US/Eastern"))
    start = (end - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    url = (
        f"https://huggingface.co/api/daily_papers"
        f"?date={start}"
    )
    
    # Query API
    try:
        response = requests.get(
            url,
            cookies={"session": str(uuid.uuid4())},
            headers={"Accept": "*/*", "User-Agent": request.headers.get("User-Agent")},
        )
        response.raise_for_status()

        quotes = json.loads(response.content.decode('utf-8'))
        flatten_data = json_normalize(quotes)
        
        title = flatten_data.title
        published_at = pd.to_datetime(flatten_data.publishedAt).dt.date
        submittedBy = flatten_data['submittedBy.fullname']
        summary = flatten_data['paper.summary']
        upvotes = flatten_data['paper.upvotes']
    
        papers_data=pd.concat([title,published_at,submittedBy,summary,upvotes],axis=1)
        papers_data.columns = ['Title','Published','Submitted','Summary','Upvotes']
        papers_data['Summary'] = papers_data['Summary'].str.replace('\n', '<br>')
        return papers_data.to_html(classes='table table-striped', index=False,escape=False)
    except (KeyError, IndexError, requests.RequestException, ValueError):
        return None

def remove_references_from_text(text):
    # Step 1: Remove the references section
    references_patterns = ["references", "bibliography", "works cited"]
    lower_text = text.lower()
    for pattern in references_patterns:
        ref_index = lower_text.find(pattern)
        if ref_index != -1:
            text = text[:ref_index].strip()
            break
    
    # Step 2: Remove in-text citations (simple patterns like [1], (Smith et al., 2020))
    # Pattern for [1], [12], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Pattern for (Smith et al., 2020) or similar
    text = re.sub(r'\(\w+ et al\., \d{4}\)', '', text)
    
    return text

def remove_section_from_text(text, section_heading):
    lower_text = text.lower()
    
    # Find the start of the section
    start_index = lower_text.find(section_heading.lower())
    
    if start_index == -1:
        return text  # Section not found
    
    # Find the end of the section (optional)
    next_section_pattern = r'\n\s*(introduction|methods|results|discussion|conclusion|references)\b'
    match = re.search(next_section_pattern, lower_text[start_index:])
    
    if match:
        end_index = start_index + match.start()
    else:
        end_index = len(text)  # No further sections found; remove till the end
    
    # Remove the section
    return text[:start_index].strip() + "\n" + text[end_index:].strip()

def extract_text_from_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    text = ""
    conversation_history = []
    system_message = "Use a formal tone and do not introduce yourself. Don't ask any question at the end. You are a PhD Student in Deep Learning. You need to explain the novelty in this paper."
    conversation_history.append({"role": "system", "content": system_message})
    i=0
    for page in data:
        text += page.page_content
        i+=1
        if i == 4:
            break
    # Remove the "Literature Review" section
    cleaned_text = remove_section_from_text(text, "Literature Review")
    
    # Optionally, remove the references and in-text citations as well
    cleaned_text = remove_references_from_text(cleaned_text)
    conversation_history.append({"role": "user", "content": cleaned_text})
    response = ollama.chat(
                model="llama3.1",
                messages=conversation_history,
                options = ollama.Options(context_length=8096)
            )
            
    # Extract the response content
    summary = response.get('message', {}).get('content', '').strip()
            
    return summary