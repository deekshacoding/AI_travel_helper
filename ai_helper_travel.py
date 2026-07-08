import os
import re
from typing import List

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="AI Travel Helper", page_icon="🌍", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f8f1 0%, #edf5e8 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f2f8ec 0%, #e3f0d8 100%);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2f5d3d;
    }
    .card {
        background: rgba(255,255,255,0.88);
        border: 1px solid #d9e7d3;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def search_web(query: str, max_results: int = 6) -> List[dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=max_results))
    return results


@st.cache_data(show_spinner=False)
def scrape_page(url: str, max_chars: int = 4000) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(soup.stripped_strings)
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


@st.cache_data(show_spinner=False)
def build_research_context(destination: str) -> str:
    search_queries = [
        f"{destination} travel guide",
        f"{destination} tourist attractions",
        f"{destination} hotels 4 star and above",
    ]

    collected_text = []
    seen_urls = set()
    for query in search_queries:
        for item in search_web(query, max_results=3):
            url = item.get("href")
            if url and url not in seen_urls:
                seen_urls.add(url)
                content = scrape_page(url)
                if content:
                    collected_text.append(f"Source: {url}\n{content}")

    if not collected_text:
        return f"No web data was found for {destination}."

    return "\n\n".join(collected_text[:8])


@st.cache_data(show_spinner=False)
def get_gemini_research(destination: str, web_context: str, api_key: str, model_name: str) -> str:
    if not api_key:
        return "Please enter a Gemini API key to research destinations."

    selected_model = model_name.strip() or "gemini-1.5-flash"
    llm = ChatGoogleGenerativeAI(
        model=selected_model,
        google_api_key=api_key,
        temperature=0.2,
    )
    prompt = PromptTemplate(
        template="""
You are an expert travel research assistant.
Use the web context below to write a concise travel report for {destination}.
Return the answer in Markdown with the following sections:
1. Overview
2. Top tourist destinations
3. Best time to visit
4. Travel tips

Keep the answer practical and useful for a traveler.

Web context:
{web_context}
""",
        input_variables=["destination", "web_context"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(destination=destination, web_context=web_context)


@st.cache_data(show_spinner=False)
def get_groq_chat_response(question: str, api_key: str, model_name: str) -> str:
    if not api_key:
        return "Please enter a Groq API key to use the travel assistant."

    selected_model = model_name.strip() or "llama3-8b-8192"
    llm = ChatGroq(
        model_name=selected_model,
        groq_api_key=api_key,
        temperature=0.4,
    )
    prompt = PromptTemplate(
        template="""
You are a helpful travel assistant. Answer only travel-related questions.
Keep replies short, practical, and friendly.

User question: {question}
""",
        input_variables=["question"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(question=question)


def render_landing_page() -> None:
    st.title("🌿 AI Travel Helper")
    st.caption("Plan smarter with AI research, hotel suggestions, and a travel chatbot.")

    st.markdown(
        """
        <div class="card">
        <h3>What this app does</h3>
        <ul>
            <li>Researches any destination using Gemini, DuckDuckGo, and web scraping.</li>
            <li>Highlights tourist spots
            <li>Provides a fast travel chatbot on the side using Groq.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Start planning", use_container_width=True):
        st.session_state.app_started = True
        st.rerun()


def main() -> None:
    if "app_started" not in st.session_state:
        st.session_state.app_started = False

    if "gemini_key" not in st.session_state:
        st.session_state.gemini_key = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if "groq_key" not in st.session_state:
        st.session_state.groq_key = os.getenv("GROQ_API_KEY", "")
    if "groq_model" not in st.session_state:
        st.session_state.groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    if not st.session_state.app_started:
        render_landing_page()
        return

    with st.sidebar:
        st.header("🔐 API keys")
        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            key="gemini_key",
            help="Paste your Gemini API key here or set GOOGLE_API_KEY/GEMINI_API_KEY in your environment.",
        )
        gemini_model = st.text_input(
            "Gemini model",
            value=st.session_state.get("gemini_model", "gemini-1.5-flash"),
            key="gemini_model",
            help="Example: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash",
        )
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            key="groq_key",
            help="Paste your Groq API key here or set GROQ_API_KEY in your environment.",
        )
        groq_model = st.text_input(
            "Groq model",
            value=st.session_state.get("groq_model", "llama3-8b-8192"),
            key="groq_model",
            help="Example: llama3-8b-8192, llama-3.3-70b-versatile",
        )
        st.caption("Add your keys here to enable research and chat features.")

    st.title("🌍 Travel Research")
    st.caption("Search a destination and receive a travel brief with tourist spots and hotel suggestions.")

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        destination = st.text_input("Search travel destination", placeholder="Example: Bali, Paris, Kyoto")
        if st.button("Research destination", use_container_width=True) and destination:
            with st.spinner("Researching travel details..."):
                web_context = build_research_context(destination)
                report = get_gemini_research(destination, web_context, gemini_key, gemini_model)
            st.session_state.current_report = report
            st.session_state.current_destination = destination
            st.session_state.current_context = web_context

    with col2:
        st.markdown("### 🤖 Travel assistant")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask for travel advice")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = get_groq_chat_response(question, groq_key, groq_model)
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if "current_report" in st.session_state and st.session_state.current_destination:
        st.markdown(f"### Research for {st.session_state.current_destination}")
        st.markdown(st.session_state.current_report)


if __name__ == "__main__":
    main()
