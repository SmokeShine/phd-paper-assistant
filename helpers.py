import csv
import datetime
import pytz
import requests
import urllib
import uuid
import json
from pandas import json_normalize
import pandas as pd
from flask import redirect, render_template, request, session
from functools import wraps


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


