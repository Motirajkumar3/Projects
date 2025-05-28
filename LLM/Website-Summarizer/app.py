from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

app = Flask(__name__)

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def scrape_text_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Try multiple selectors for different website structures
    selectors = [
        {"name": "div", "attrs": {"id": "mw-content-text"}},  # Wikipedia
        {"name": "article"},
        {"name": "main"},
        {"name": "div", "attrs": {"class": "content"}}
    ]

    content_div = None
    for sel in selectors:
        content_div = soup.find(**sel)
        if content_div:
            print(f"Found content using selector: {sel}")
            break

    if not content_div:
        print("Main content not found.")
        return None

    # Remove unwanted tags
    for tag in content_div.find_all(["script", "style", "table", "nav", "footer", "aside"]):
        tag.decompose()

    paragraphs = content_div.find_all("p")
    if not paragraphs:
        return None

    text = " ".join(p.get_text(strip=True) for p in paragraphs)
    return text

def generate_summary(text, max_chars=1024):
    text = text[:max_chars]  # Limit input size for the model
    try:
        summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Error generating summary."

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    url = ""
    message = ""
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            text = scrape_text_from_url(url)
            if text:
                summary = generate_summary(text)
            else:
                message = "Could not extract readable content from the page. Try a different site or check the URL."
        else:
            message = "Please enter a valid URL."
    return render_template("index.html", summary=summary, url=url, message=message)

if __name__ == "__main__":
    app.run(debug=True)
