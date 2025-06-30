from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from your_rag_module import load_faiss_and_data, rag_query
from youtubesearchpython import VideosSearch

app = FastAPI()

# CORS Middleware (allow all during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for /recommend


class RecommendRequest(BaseModel):
    genre: str
    rating: float
    actor: Optional[str] = None


# Load embeddings and models
print("🔁 Loading FAISS, data, model, GPT client...")
faiss_index, final_df, model, client = load_faiss_and_data()
print("✅ Backend ready.")

# /recommend RAG movie endpoint


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        result = rag_query(
            genre=req.genre,
            min_rating=req.rating,
            actor=req.actor,
            top_k=5000,
            faiss_index=faiss_index,
            final_df=final_df,
            model=model,
            client=client
        )
        return {"results": result}
    except Exception as e:
        return {"error": str(e)}

# /trailer route to fetch YouTube videoId


@app.get("/trailer")
def get_trailer(title: str = Query(..., description="Movie title to search")):
    try:
        print(f"🔎 Searching YouTube for: {title} trailer")
        search = VideosSearch(f"{title} trailer", limit=1)
        results = search.result()
        print("🧠 YouTube API raw result:", results)  # <-- ADD THIS LINE

        if not results["result"]:
            print("❌ No trailer found.")
            return {"videoId": "YoHD9XEInc0"}

        video_id = results["result"][0].get("id", "YoHD9XEInc0")
        print(f"✅ Found videoId: {video_id}")
        return {"videoId": video_id}
    except Exception as e:
        print(f"❌ Trailer fetch error: {e}")
        return {"videoId": "YoHD9XEInc0"}