# 🌐 Website Text Summarizer App

This Flask web app allows users to enter the URL of any webpage (especially article-based pages like Wikipedia) and receive a concise summary of the page's main content using a transformer-based summarization model.

## 🚀 Features

- Extracts meaningful content from websites (Wikipedia, blogs, news sites, etc.)
- Cleans navigation, ads, and unwanted page elements
- Summarizes the content using `facebook/bart-large-cnn` from Hugging Face Transformers
- Dark-themed, mobile-friendly interface

## 🧰 Technologies Used

- Python 🐍
- Flask 🌶️
- BeautifulSoup 🥣
- Hugging Face Transformers 🤗 (`facebook/bart-large-cnn`)
- HTML + CSS (Dark Theme UI)

## 📦 Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/website-summarizer.git
   cd website-summarizer
