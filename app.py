import gradio as gr
import requests
import fitz
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from faster_whisper import WhisperModel
import os

# =========================
# INITIALIZE MODELS
# =========================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
whisper_model = WhisperModel("base", compute_type="int8")

# Retrieve Groq API key from environment variables
groq_api_key = "YOUR_API"
client = Groq(api_key=groq_api_key)
MODEL_NAME = "llama-3.3-70b-versatile"

# Global storage
sections = {}
section_texts = []
index = None


# =========================
# PDF FUNCTIONS
# =========================

def download_arxiv_pdf(arxiv_id):
    try:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(url)
        response.raise_for_status()

        file_path = f"{arxiv_id}.pdf"
        with open(file_path, "wb") as f:
            f.write(response.content)

        return file_path
    except:
        return None


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_sections(text):

    patterns = [
        r"\n([IVX]+\.\s+[A-Z][A-Z\s]+)",        # Roman numeral ALL CAPS
        r"\n(\d+\.\s+[A-Z][^\n]+)",             # 1. Introduction
        r"\n(\d+\s+[A-Z][^\n]+)",               # 1 Introduction
        r"\n([A-Z][A-Z\s]{3,})\n"               # ALL CAPS standalone
    ]

    matches = []
    for pattern in patterns:
        matches.extend(list(re.finditer(pattern, text)))

    matches = sorted(matches, key=lambda x: x.start())

    sections = {}
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        sections[title] = text[start:end].strip()

    return sections


# =========================
# VECTOR STORE
# =========================

def build_vector_store(sections_dict):
    global index, section_texts

    section_texts = list(sections_dict.values())

    if len(section_texts) == 0:
        index = None
        return

    embeddings = embedding_model.encode(section_texts)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)


# =========================
# LOAD PAPER
# =========================

def load_paper(arxiv_id):
    global sections, index

    pdf_path = download_arxiv_pdf(arxiv_id)

    if pdf_path is None:
        return gr.update(choices=[]), "❌ Invalid arXiv ID"

    text = extract_text_from_pdf(pdf_path)
    sections = extract_sections(text)

    build_vector_store(sections)

    return gr.update(choices=list(sections.keys())), "✅ Paper Loaded Successfully"


# =========================
# SUMMARIZATION
# =========================

def summarize_section(section_title):
    if section_title not in sections:
        return "Please load paper first."

    content = sections[section_title]

    prompt = f"""
You are an expert AI research assistant.

Generate a structured scientific summary:
- Main idea
- Key technical concepts
- Important equations explained simply
- Why this section matters

Section Title: {section_title}
Section Content:
{content[:6000]}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# =========================
# RAG CHAT
# =========================

def rag_chat(message, history):
    global index

    if index is None:
        history.append((message, "Please load a paper first."))
        return history, ""

    query_embedding = embedding_model.encode([message])
    query_embedding = np.array(query_embedding).astype("float32")

    D, I = index.search(query_embedding, k=3)

    retrieved = "\n\n".join([section_texts[i] for i in I[0]])

    prompt = f"""
Answer strictly using the provided research paper context.
If the answer is not found, say:
"The answer is not available in the provided paper."

Context:
{retrieved}

Question:
{message}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    history.append((message, answer))
    return history, ""


# =========================
# VOICE CHAT
# =========================

def voice_chat(audio, history):
    if audio is None:
        return history, ""

    segments, _ = whisper_model.transcribe(audio)
    text = "".join([segment.text for segment in segments])

    return rag_chat(text, history)


# =========================
# GRADIO UI
# =========================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 ArXiv RAG Research Assistant")

    with gr.Row():
        arxiv_input = gr.Textbox(label="Enter arXiv ID (e.g., 1706.03762)")
        load_button = gr.Button("Load Paper")

    load_status = gr.Markdown()

    section_dropdown = gr.Dropdown(label="Select Section")
    summarize_button = gr.Button("Generate Summary")
    summary_output = gr.Markdown()

    gr.Markdown("## 💬 Research Chat")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask a question")
    send = gr.Button("Send")

    gr.Markdown("## 🎙 Voice Question")
    audio_input = gr.Audio(type="filepath")
    voice_button = gr.Button("Ask via Voice")

    # Actions
    load_button.click(load_paper, inputs=arxiv_input, outputs=[section_dropdown, load_status])
    summarize_button.click(summarize_section, inputs=section_dropdown, outputs=summary_output)
    send.click(rag_chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
    voice_button.click(voice_chat, inputs=[audio_input, chatbot], outputs=[chatbot, msg])

demo.launch(debug=True)
