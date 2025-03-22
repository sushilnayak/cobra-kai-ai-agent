import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from victor_ai import VictorAI

# Load environment variables
load_dotenv()

# Initialize the Victor AI agent
victor_ai = VictorAI()

# Create FastAPI app
app = FastAPI(
    title="Victor AI API",
    description="REST API for Kobra Kai Karate School Franchise AI Assistant",
    version="1.0.0"
)

# Add CORS middleware to allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your frontend domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request/response models
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str
    follow_ups: List[str]


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ConversationRequest(BaseModel):
    messages: List[Message]


class ConversationResponse(BaseModel):
    response: str
    follow_ups: List[str]


class StatusResponse(BaseModel):
    status: str
    message: str
    vertex_ai_connected: bool


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    formatted_response: Dict[str, Any] = Field(default_factory=dict,
                                               description="Structured content for better rendering")
    follow_up_suggestions: List[str]


# API endpoints
@app.get("/", response_model=StatusResponse)
async def root():
    """Get API status and check if Vertex AI is connected"""
    return {
        "status": "online",
        "message": "Victor AI API is running. Use /query or /conversation endpoints to interact with the agent.",
        "vertex_ai_connected": victor_ai.llm is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Simple query endpoint - send a single question and get a response
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Generate response
    response = victor_ai.generate_response(request.query)

    # Generate follow-up questions
    follow_ups = victor_ai.suggest_follow_ups(request.query, response)

    return {
        "response": response,
        "follow_ups": follow_ups
    }


@app.post("/conversation", response_model=QueryResponse)
async def conversation(request: ConversationRequest):
    """
    Conversation endpoint - send a list of messages and get a response
    This allows for maintaining context across multiple turns of conversation
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    # Extract the latest user query (last user message)
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user messages found")

    latest_user_query = user_messages[-1].content

    # Format conversation history as context (optional enhancement)
    # This could be improved to use proper conversation history
    conversation_history = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages[:-1]])

    # Generate response
    response = victor_ai.generate_response(latest_user_query)

    # Generate follow-up questions
    follow_ups = victor_ai.suggest_follow_ups(latest_user_query, response)

    return {
        "response": response,
        "follow_ups": follow_ups
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "vertex_connection": "connected" if victor_ai.llm is not None else "disconnected"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Simple chat endpoint for the React frontend
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Generate response
    response = victor_ai.generate_response(request.message)

    # Process response to create formatted content
    formatted_response = process_response_for_frontend(response)

    # Generate follow-up questions
    follow_ups = victor_ai.suggest_follow_ups(request.message, response)

    return {
        "response": response,
        "formatted_response": formatted_response,
        "follow_up_suggestions": follow_ups
    }


def process_response_for_frontend(text):
    """
    Process the response to create a structured format for the frontend
    Returns a dict with formatted content sections
    """
    # Simple processing for now - split by paragraphs and identify potential sections
    paragraphs = [p for p in text.split('\n') if p.strip()]

    result = {
        "sections": []
    }

    current_section = {"title": "", "content": []}

    for p in paragraphs:
        # Check if this is a header/section title (simple heuristic)
        if len(p) < 50 and (p.endswith(':') or p.isupper() or p.startswith('#')):
            # If we have content in the current section, add it to results
            if current_section["content"]:
                result["sections"].append(current_section)
                current_section = {"title": p.strip('# :'), "content": []}
            else:
                current_section["title"] = p.strip('# :')
        else:
            # Regular paragraph
            current_section["content"].append(p)

    # Add the last section if it has content
    if current_section["content"]:
        result["sections"].append(current_section)

    # If no structured sections were found, create a default one
    if not result["sections"]:
        result["sections"] = [{"title": "", "content": paragraphs}]

    return result


# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn

    # Use environment variables or default values for host and port
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)
