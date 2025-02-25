import os
import re
import pdfplumber
import requests
import numpy as np
import markdown
from flask import Flask, render_template_string, request
from dotenv import load_dotenv
load_dotenv()

from together import Together  # Import the together.ai client
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))            # Instantiate the client

from sentence_transformers import SentenceTransformer

###################
# CONFIG & GLOBALS
###################

app = Flask(__name__)

PDF_PATH = "input.pdf"
EMBEDDED_CLAUSES = []  # will store (clause_title, clause_text, embedding_vector)
EMBEDDING_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

# We'll also specify which ChatCompletion model to use for final answers.
#CHAT_MODEL = "gpt-3.5-turbo"

###############################
# 1. PDF TEXT EXTRACTION      #
###############################
def extract_text_from_pdf(pdf_path):
    """
    Reads all pages from the PDF and returns a single string.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

###########################
# 2. CLAUSE-BASED CHUNKING
###########################
def chunk_text_by_clause(full_text):
    """
    Splits the PDF text by headings such as 'Clause 1', 'Clause 2', 'Schedule 3', or a leading number.
    Returns a list of (clause_title, clause_text).
    """
    # Regex to match lines that look like "Clause 9", "Schedule 3", "1. Definitions", etc.
    # Adjust this pattern to fit your PDF's structure more precisely.
    clause_pattern = re.compile(r'^\s*(Clause\s+\d+|Schedule\s+\d+|\d+\.\s+.*)', re.IGNORECASE)

    lines = full_text.split('\n')
    chunks = []
    current_title = "Intro/Preamble"
    current_lines = []

    for line in lines:
        # If this line looks like a clause heading
        if clause_pattern.match(line.strip()):
            # If we already have some lines collected, store them as a chunk
            if current_lines:
                combined_text = "\n".join(current_lines).strip()
                chunks.append((current_title, combined_text))
                current_lines = []
            current_title = line.strip()
        else:
            current_lines.append(line)

    # Add the last chunk
    if current_lines:
        combined_text = "\n".join(current_lines).strip()
        chunks.append((current_title, combined_text))

    return chunks

################################
# 3. LOCAL EMBEDDING & STORAGE #
################################
def load_local_embedding_model():
    print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

def embed_clauses_by_title(clause_pairs, model):
    """
    Given a list of (clause_title, clause_text) and a SentenceTransformer model,
    returns a list of (clause_title, clause_text, embedding_vector).
    """
    print("Embedding clauses locally... This may take a bit for large PDFs.")
    embedded_data = []
    titles = []
    texts = []

    # We'll embed each chunk as 'title + text' to capture context
    for title, text in clause_pairs:
        combined = f"{title}\n{text}"
        titles.append(title)
        texts.append(text)

    # Actually embed all combined strings in one batch
    combined_strs = [f"{t}\n{x}" for (t,x) in clause_pairs]
    embeddings = model.encode(combined_strs, batch_size=8, show_progress_bar=True)

    for (title, text), vector in zip(clause_pairs, embeddings):
        embedded_data.append((title, text, vector))

    return embedded_data

##################################
# 4. RETRIEVAL VIA COSINE SIM    #
##################################
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_clauses(user_query, embedded_clauses, embedding_model, top_k=3):
    """
    1. Embed the user_query with the local model.
    2. Compute cosine similarity with each clause vector.
    3. Sort & return top_k as (clause_title, clause_text).
    """
    query_embedding = embedding_model.encode([user_query])[0]
    scored = []
    for (title, text, vector) in embedded_clauses:
        score = cosine_similarity(query_embedding, vector)
        scored.append((score, title, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_matches = scored[:top_k]
    # Return as a list of (title, text)
    result = [(t, c) for (_, t, c) in top_matches]
    return result

#############################
# 5. CALLING THE LOCAL LLM
#############################
def ask_together(context, user_query):
    """
    Uses the together.ai API with the meta-llama model to answer the query based on provided context.
    """
    # Revised prompt: explicitly instruct the model to output only the final answer.
    prompt = (
        # "You are a legal/contractual Q&A assistant. Using only the following text, provide a concise final answer to the user's question. "
        # "Do NOT include any internal chain-of-thought or reasoning steps in your output. "
        # "Your output should contain only the final answer as a single clear paragraph with references to relevant clauses if applicable.\n\n"
        "You are a legal/contractual Q&A assistant. You have access to the following text. "\
        "Answer the user's question based ONLY on this text. If you're unsure, say so. "\
        "Provide references to relevant clauses or schedules ONLY if they explicitly appear in the text. "\
        "Use the following formatting guidelines:\n"\
        "## Start headings with '##'\n"\
        "- Use a bulleted list with dashes ('-') for subpoints\n"\
        "- Bold all clause references, e.g., **(Clause X.Y)**\n"\
        "- Include an extra line break after each bullet group.\n"\
        "- Italicize any disclaimers or notes.\n"\
        "Use actual HTML tags for formatting:\n"
        "- <strong> for bold\n"
        "- <em> for italics\n"
        "- <u> for underline\n"
        "If HTML rendering is supported, you may use <u>...</u> for underlining specific text.\n"
        "Relevant text:\n" + context + "\n\n"
        "User Query: " + user_query + "\n\n"
        "Final Answer:"
    )
    
    messages = [{"role": "system", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",  # Updated model name
            messages=messages,
            max_tokens=1500,
            temperature=0,  # Adjust temperature as needed
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=True
        )
        
        generated_text = ""
        for token in response:
            if hasattr(token, 'choices'):
                generated_text += token.choices[0].delta.content
        print("Generated Answer:", generated_text)
        return generated_text

    except Exception as e:
        print("Error calling together.ai API:", e)
        return "Sorry, I couldn't process your request."





###################################
# 6. FLASK APP ROUTES & TEMPLATES #
###################################
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lactalis Warehousing Agreement Chatbot</title>
    <style>
        :root {
            --background-color: #FFFFFF;
            --text-color: #1A1A1A;
            --border-color: #E5E5E5;
            --input-background: #FFFFFF;
            --button-color: #5E44FF;
            --button-hover: #4B37CC;
            --human-background: #F8F8F8;
            --assistant-background: #FFFFFF;
            --code-background: #F8F8F8;
            --welcome-background: #F0F4FF;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--background-color);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .chat-container {
            flex-grow: 1;
            overflow-y: auto;
            margin-bottom: 120px;
        }
        .welcome-container {
            padding: 48px 20%;
            background-color: var(--welcome-background);
            text-align: center;
            border-bottom: 1px solid var(--border-color);
        }
        .welcome-content {
            max-width: 800px;
            margin: 0 auto;
        }
        .welcome-title {
            font-size: 32px;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 16px;
        }
        .welcome-subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 24px;
            line-height: 1.6;
        }
        .welcome-features {
            display: flex;
            justify-content: center;
            gap: 32px;
            margin-top: 32px;
        }
        .feature-item {
            flex: 1;
            max-width: 240px;
            padding: 24px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        .feature-icon {
            font-size: 24px;
            margin-bottom: 16px;
            color: var(--button-color);
        }
        .feature-title {
            font-weight: 600;
            margin-bottom: 8px;
        }
        .feature-description {
            font-size: 14px;
            color: #666;
        }
        .message-wrapper {
            padding: 0 20%;
            border-bottom: 1px solid var(--border-color);
        }
        @media (max-width: 1024px) {
            .message-wrapper, .welcome-container {
                padding: 24px 5%;
            }
            .welcome-features {
                flex-direction: column;
                align-items: center;
            }
            .feature-item {
                width: 100%;
                max-width: none;
            }
        }
        .message {
            max-width: 800px;
            margin: 0 auto;
            padding: 24px 0;
        }
        .human-message {
            background-color: var(--human-background);
        }
        .assistant-message {
            background-color: var(--assistant-background);
        }
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 24px 20%;
            background-color: var(--background-color);
            border-top: 1px solid var(--border-color);
        }
        form {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-width: 800px;
            margin: 0 auto;
            width: 100%;
            position: relative;
        }
        .query-box {
            width: 100%;
            padding: 16px;
            font-size: 16px;
            line-height: 1.5;
            color: var(--text-color);
            background-color: var(--input-background);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            resize: none;
            max-height: 200px;
            outline: none;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }
        .query-box:focus {
            border-color: var(--button-color);
            box-shadow: 0 0 0 2px rgba(94, 68, 255, 0.1);
        }
        .submit-button {
            position: absolute;
            right: 12px;
            bottom: 12px;
            background-color: var(--button-color);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .submit-button:hover {
            background-color: var(--button-hover);
        }
        .message-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 8px;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        {% if not answer %}
        <div class="welcome-container">
            <div class="welcome-content">
                <h1 class="welcome-title">Welcome to NewCold Legal Support</h1>
                <p class="welcome-subtitle">Your AI-powered assistant for legal queries and document analysis. How can I help you today?</p>
                <div class="welcome-features">
                    <div class="feature-item">
                        <div class="feature-icon">📋</div>
                        <div class="feature-title">Document Analysis</div>
                        <div class="feature-description">Get instant insights on legal documents and agreements</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">⚖️</div>
                        <div class="feature-title">Legal Guidance</div>
                        <div class="feature-description">Clear explanations of legal terms and concepts</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">🔍</div>
                        <div class="feature-title">Quick Answers</div>
                        <div class="feature-description">Fast responses to your legal queries</div>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        {% if answer %}
        <!-- Human Message -->
        <div class="message-wrapper human-message">
            <div class="message">
                <div class="message-label">Human</div>
                <p>{{ query }}</p>
            </div>
        </div>
        <!-- Final Answer -->
        <div class="message-wrapper assistant-message">
            <div class="message">
                <div class="message-label">NewCold Assistant</div>
                <p>{{ answer|safe }}</p>
            </div>
        </div>
        {% endif %}
    </div>
    <div class="input-container">
        <form method="POST" action="/">
            <textarea 
                id="query" 
                name="query" 
                rows="1" 
                class="query-box" 
                placeholder="Ask anything about NewCold legal queries..."
                oninput="this.style.height = 'auto'; this.style.height = this.scrollHeight + 'px';"
            >{{ query }}</textarea>
            <button type="submit" class="submit-button">Send</button>
        </form>
    </div>
</body>
</html>


"""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_query = request.form.get("query", "")
        if not user_query.strip():
            answer = "Please enter a valid query."
        else:
            top_clauses = find_similar_clauses(user_query, EMBEDDED_CLAUSES, EMBEDDING_MODEL, top_k=3)
            context_parts = [f"*** {title} ***\n{text}" for (title, text) in top_clauses]
            final_context = "\n\n".join(context_parts)
            # Use the together.ai API instead of the local model
            raw_answer = ask_together(final_context, user_query)
            answer = markdown.markdown(raw_answer)
        return render_template_string(HTML_PAGE, query=user_query, answer=answer)
    return render_template_string(HTML_PAGE, query="", answer="")

########################
# 7. APP STARTUP       #
########################
EMBEDDING_MODEL = None
EMBEDDED_CLAUSES = []

def load_and_embed_pdf():
    global EMBEDDING_MODEL, EMBEDDED_CLAUSES

    # 1. Load local embedding model
    EMBEDDING_MODEL = load_local_embedding_model()

    # 2. Extract text from PDF
    raw_text = extract_text_from_pdf(PDF_PATH)

    # 3. Clause-based chunking
    clause_pairs = chunk_text_by_clause(raw_text)
    print(f"Found {len(clause_pairs)} clause-level chunks from PDF.")

    # 4. Embed each clause
    EMBEDDED_CLAUSES = embed_clauses_by_title(clause_pairs, EMBEDDING_MODEL)
    print(f"Embedded {len(EMBEDDED_CLAUSES)} clauses.")

load_and_embed_pdf()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    #load_and_embed_pdf()
    app.run(debug=False, host="0.0.0.0", port=port)