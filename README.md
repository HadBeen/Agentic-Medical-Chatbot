# Medical AI chatbot

An AI-powered educational medical assistant built with **LangGraph, Pinecone, and FastAPI**, combining retrieval-augmented generation (RAG) with multimodal capabilities to provide context-aware medical information from a historical medical knowledge base.

The system retrieves relevant information from the Gale Encyclopedia of Medicine and uses an AI agent to generate responses, with optional Tavily web search for external information.

## Key Features

- **RAG Pipeline:** Retrieves relevant medical information from an indexed medical encyclopedia using Pinecone.
- **LangGraph Agent:** Orchestrates retrieval, routing, external search, and response generation.
- **Vector Database:** Stores and searches medical document embeddings using Pinecone.
- **Web Search Fallback:** Integrates Tavily to retrieve external information when needed.
- **Multimodal Support:** Accepts image observations and provides AI-powered descriptions.
- **Speech Integration:** Supports voice input through Whisper and text-to-speech output.
- **REST API:** Exposes application functionality through FastAPI.
- **Interactive UI:** Provides a web interface built with vanilla JavaScript, HTML, and CSS.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| AI & LLMs | OpenAI API, Whisper |
| Agent Orchestration | LangGraph, LangChain |
| RAG & Embeddings | OpenAI Embeddings |
| Vector Database | Pinecone |
| External Search | Tavily |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Document Processing | PDF Processing |

## Architecture

User Query → LangGraph Agent → Knowledge Retrieval (Pinecone) → Routing → Tavily Search (if needed) → LLM Response → User

