import os

import requests
from flask import Flask, flash, redirect, render_template, request, session, jsonify
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from cs50 import SQL
from helpers import apology, login_required, lookup_titles
import ollama

# Configure application
app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")


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
    """Show portfolio of stocks"""
    # there is no index.html
    # this needs to be populated after register
    # which table stores the stock owned by the user. maybe it is going in the file system?
    # too much memory usage. unlikely
    # Add one or more new tables to finance.db via which to keep track of the purchase.
    # it is going in db. so, I need a script to create a table
    #     CREATE TABLE transactions (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, user_id INTEGER NOT NULL,company TEXT NOT NULL, qty NUMERIC NOT NULL, price NUMERIC NOT NULL, transaction_type text NOT NULL);
    # CREATE UNIQUE INDEX purchase_index ON purchase (id);
    # this has no post/get

    # is there a corresponding table in layout.html
    # birthday was passed as a dict
    papers_data = lookup_titles()
    if len(papers_data) == 0:
        return apology("No Data", 200)
    return render_template("index.html", papers_data=papers_data), 200


@app.route("/history")
@login_required
def history():
    """Show history of transactions"""
    # so this must be stored somewhere. is it db?
    # sqlite> .schema
    # CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, username TEXT NOT NULL, hash TEXT NOT NULL, cash NUMERIC NOT NULL DEFAULT 10000.00);
    # CREATE TABLE sqlite_sequence(name,seq);
    # where is this used? this is auto index
    # CREATE UNIQUE INDEX username ON users (username);
    # so there are 3 tables. all are empty
    # do i need to filter it for a user - how will i know the user
    transaction_data = db.execute(
        """SELECT company,qty,price,transaction_type FROM transactions WHERE user_id = ? """,
        session["user_id"],
    )
    return render_template("history.html", transaction_data=transaction_data), 200


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
            "SELECT * FROM users WHERE username = ?", request.form.get("username")
        )

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(
            rows[0]["hash"], request.form.get("password")
        ):
            return apology("invalid username and/or password", 400)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/"), 200

    # User reached route via GET (as by clicking a link or via redirect)
    elif request.method == "GET":
        return render_template("login.html")
    else:
        return redirect("/"), 200


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
            return redirect("/"), 200
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
    system = "You are a helpful assistant that provides simple and easy-to-understand explanations for complex topics. Your explanations can contain technical jargon to make the concepts clear."
    
    if selected_text:
        # Add system message if it's the start of a new conversation
        if not conversation_history:
            conversation_history.append({"role": "system", "content": system})
        
        # Add user's message to history
        conversation_history.append({"role": "user", "content": selected_text})

        # Get the response from the model
        response = ollama.chat(
            model="llama3.1",
            messages=conversation_history
        )

        # Extract the response content
        explanation = response['message']['content'].strip()
        
        # Add the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": explanation})
        
        return jsonify({"explanation": explanation})

    return jsonify({"explanation": "No text provided"}), 400