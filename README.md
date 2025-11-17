# AI Voice Support Assistant — LiveKit + Gemini Realtime + RAG

A high-performance, real-time AI voice assistant built using **LiveKit Agents**, **Google Gemini**, and a custom **Retrieval-Augmented Generation (RAG)** pipeline.

This project enables users to join a LiveKit audio room, speak naturally with an AI assistant, and receive intelligent, context-aware answers grounded in your own knowledge base. It serves as a powerful starter kit for building AI call center agents, support bots, or any real-time voice-first application.

---

## ✨ Key Features

* **Real-time Conversation:** Sub-200ms audio latency using LiveKit's WebRTC data plane for seamless, natural conversation.
* **Interruptible AI:** The agent can be interrupted at any time, just like a real person, thanks to LiveKit's streaming STT/TTS capabilities.
* **Context-Aware RAG:** The assistant doesn't just guess; it retrieves information from a local vector database (ChromaDB) to provide accurate, factual answers based on your documents.
* **Advanced RAG Pipeline:** Includes a sophisticated retrieval flow with sentence-transformer embeddings and a cross-encoder re-ranker for maximum relevance.
* **Scalable Backend:** Uses FastAPI for efficiently issuing LiveKit room tokens.
* **Modern Frontend:** A clean React UI for joining and interacting with the voice room.

---

## 🛠️ Technology Stack

* **Real-time Audio/Video:** **LiveKit** (Agents SDK, React SDK)
* **AI Reasoning & STT/TTS:** **Google Gemini** (Realtime API)
* **RAG Pipeline:**
  * **Vector Database:** **ChromaDB**
  * **Embeddings:** `nomic-embed-text` (via `ollama`)
  * **Re-ranking:** `ms-marco-MiniLM-L-6-v2` (via `cross-encoder`)
* **Backend Server:** **FastAPI**
* **Frontend UI:** **React**

---

## 📡 System Architecture & Data Flow

This system coordinates several components to create a seamless voice experience.

### End-to-End Data Flow

1. **Token Request:** The **React Frontend** asks the **FastAPI Backend** for a connection token.
2. **Token Generation:** The `fastapi_room_endpoint.py` server generates a valid LiveKit JWT token, granting the user permission to join a specific room.
3. **Room Connection:** The user and the **LiveKit Agent** (`agent.py`) both connect to the same LiveKit room.
4. **Audio Streaming:** The user speaks. Their audio is streamed in real-time to the LiveKit Agent.
5. **AI Processing:** The agent forwards this audio stream to the **Gemini Realtime API**, which handles streaming Speech-to-Text (STT) and conversational reasoning.
6. **Tool Call Trigger:** When the user asks a question requiring specific knowledge (e.g., "What is your refund policy?"), Gemini's reasoning engine triggers a **Tool Call** defined in the agent.
7. **RAG Pipeline Execution:** This tool call executes the local RAG pipeline to fetch relevant context. (See deep-dive below).
8. **Grounded Response:** The retrieved context is "stuffed" back into the Gemini prompt. Gemini then generates a factually-grounded answer.
9. **Audio Response:** This text response is streamed to a Text-to-Speech (TTS) service, and the resulting audio is streamed *back* to the LiveKit room for the user to hear.

---

### 🔍 RAG Pipeline Deep-Dive

The RAG pipeline ensures that the AI's answers are accurate and based on your provided data (e.g., `rag_ready_faqs_data.jsonl`).

**Flow:**

1. **Embed Query:** The user's transcribed question is converted into a vector embedding.
2. **Vector Search:** The `knowledge_retriever.py` performs a fast vector search in **ChromaDB** to find the top-k relevant document chunks.
3. **Re-rank Results:** These initial results are passed to the `cross_encoder_reranker.py`. The cross-encoder is more computationally expensive but much more accurate at scoring the *true relevance* of each chunk to the specific query.
4. **Inject Context:** The top-ranked, highly-relevant documents are selected and injected directly into the Gemini prompt as context.

---

## 📁 Project Structure

### Backend

```plaintext
|- data
|    - rag_ready_faqs_data.jsonl
|- kb_chroma_db
|- rag
|    - config.py
|    - cross_encoder_reranker.py
|    - decode_placeholders.py
|    - embedding_wrapper.py
|    - knowledge_retriever.py
|    - loader.py
|- .env.local
|- agent.py
|- fastapi_room_endpoint.py
|- requirements.txt
```

### Frontend

```plaintext
|- node_modules
|- src
|    - components
|       - LiveKitModal.jsx
|       - SimpleVoiceAssistant.css
|       - SimpleVoiceAssistant.jsx
|    - App.css
|    - App.jsx
|    - index.css
|    - main.jsx
|- env
```

#### Install Required Dependencies

```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install @livekit/components-react @livekit/components-styles livekit-client --save
npm run dev

```

## Run the code

* Run Agent:

```bash
 python agent.py start
 ```

* Run Token Endpoint for Session Creation

 ```bash
 python  python fastapi_room_endpoint.py
 ```

* Run the Frontend React Server

```bash
npm run dev
```

### Video Preview

[![Watch the demo](https://img.youtube.com/vi/-w5pvu51PGI/0.jpg)](https://www.youtube.com/watch?v=-w5pvu51PGI)

### Helpful Sources

#### LiveKit and Gemini

* [LiveKit Website](https://cloud.livekit.io/projects/p_52xf4tddgiq/settings/keys)
* [LiveKit Playground Website](https://agents-playground.livekit.io/#cam=1&mic=1&screen=1&video=1&audio=1&chat=1&theme_color=cyan)
* [Realtime Gemini API Implementation](https://docs.livekit.io/agents/models/realtime/plugins/gemini/)
* [Agent Core Logic Implementation](https://github.com/JinzoTun/livekit-agent-python/blob/main/README.md)

#### frontend integration

* [React With LiveKit](https://docs.livekit.io/home/quickstarts/react/)
* [Yourube Tutorial](https://www.youtube.com/watch?v=Ew7fOQpkKBw)
* [Github Repo](https://github.com/techwithtim/LiveKit-AI-Car-Call-Centre/tree/main/frontend)
