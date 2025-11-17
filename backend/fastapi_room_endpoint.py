import os
from livekit import api
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(".env.local")

app = FastAPI()

# Allow CORS for frontend dev server(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load from .env.local
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


@app.get("/token")
def get_token(identity: str = "user", user: str | None = None):
    if user:
        identity = user

    room_name = f"support-{identity}"

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room=room_name, room_join=True))
        .to_jwt()
    )

    return {"token": token, "room": room_name}



if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI app on localhost:8000
    uvicorn.run("fastapi_room_endpoint:app", host="127.0.0.1", port=8000, reload=True)
    # http://127.0.0.1:8000/token?user=most

