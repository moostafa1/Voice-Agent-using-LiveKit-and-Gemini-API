import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import silero
from livekit.plugins.google import realtime

from rag.loader import load_or_create_vectorstore
from rag.knowledge_retriever import retrieve_and_rerank_knowledge

# -------------------
# Logging & Environment
# -------------------
load_dotenv(dotenv_path=".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------
# Tool: Knowledge Base Search
# -------------------
@function_tool
async def search_knowledge_base(question: str) -> str:
    """
    Tool for searching any hostsailor related topics, including technical questions, FAQs, and general hosting info.
    also can handle questions regarding to the creator of the project.
    """
    global vectorstore  # use the module-level vectorstore
    logger.info(f"Tool 'search_knowledge_base' called with question: '{question}'")

    if not vectorstore:
        return "Knowledge base is not available."

    retrieved_answer = retrieve_and_rerank_knowledge(question, vectorstore)

    if isinstance(retrieved_answer, dict):
        return retrieved_answer.get(
            "formatted_context",
            retrieved_answer.get("answer", "No answer could be found.")
        )
    return "No valid answer returned from knowledge base."

# -------------------
# LiveKit Entry Point
# -------------------
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    await ctx.wait_for_participant()
    logger.info("Connected to LiveKit room.")

    # Load or create vectorstore
    vectorstore = load_or_create_vectorstore()
    if not vectorstore or not hasattr(vectorstore, "similarity_search_with_score"):
        logger.error("Vectorstore could not be loaded or is invalid.")
        return

    agent = Agent(
        instructions="""
            You are a friendly voice assistant built by HostSailor.
            Greet users first. answer technical questions especially hosting ones using the `search_knowledge_base` tool.
            Do not assume answers without user request.
        """,
        tools=[search_knowledge_base],
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        llm=realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-09-2025",
        ),
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(
        instructions="Greet the user and ask what they need help with in a friendly tone."
    )

# -------------------
# Run CLI App
# -------------------
if __name__ == "__main__":
    vectorstore = load_or_create_vectorstore()

    if vectorstore and hasattr(vectorstore, "similarity_search_with_score"):
        logger.info(f"Vectorstore loaded successfully: {type(vectorstore)}")
    else:
        logger.error("Vectorstore could not be loaded or is invalid.")
        exit(1)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
