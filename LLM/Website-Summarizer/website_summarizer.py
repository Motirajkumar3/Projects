import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline

# Download punkt tokenizer once
nltk.download('punkt')

def fetch_text_from_url(url):
    print(f"Fetching and summarizing content from: {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text from <p> tags only
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

def clean_text(text):
    # Remove multiple spaces and new lines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common Wikipedia footer/navigation words (you can add more if needed)
    unwanted_phrases = [
        'Jump to navigation', 'Jump to search', 'References', 
        'External links', 'See also', 'Navigation menu'
    ]
    for phrase in unwanted_phrases:
        text = text.replace(phrase, '')
    return text.strip()

def chunk_text(text, max_sentences=20):
    # Correct: removed language='english' param to avoid LookupError
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = ' '.join(sentences[i:i + max_sentences])
        chunks.append(chunk)
    return chunks

def summarize_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    chunks = chunk_text(text)
    summary = ''
    for chunk in chunks:
        # Summarize each chunk and concatenate
        summary_chunk = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summary += summary_chunk + ' '
    return summary.strip()

if __name__ == "__main__":
    url = input("Enter the URL of the Wikipedia article: ").strip()
    raw_text = fetch_text_from_url(url)
    clean = clean_text(raw_text)
    summary = summarize_text(clean)
    print(f"\nSummary for URL: {url}\n")
    print(summary)
