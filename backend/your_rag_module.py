# backend/rag_module.py

import pandas as pd
import numpy as np
import faiss
import ast
import pickle
import os
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


def load_faiss_and_data():
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    faiss_index = faiss.read_index("faiss_movie_index.index")

    with open("id_mapping.pkl", "rb") as f:
        _ = pickle.load(f)

    df = pd.read_csv("cleaned_movies.csv")
    df['actors'] = df['actors'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['genres_parsed'] = df['genres_parsed'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.getenv("OPENAI_API_KEY")
)

    return faiss_index, df, model, client


def rag_query(genre, min_rating, actor, top_k, faiss_index, final_df, model, client):
    query = f"I want a {genre} movie rated above {min_rating}"
    if actor:
        query += f" starring {actor}"
    query += " that I might enjoy."

    query_embedding = model.encode([query])
    query_embedding = np.array(
        query_embedding) / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    D, I = faiss_index.search(query_embedding, top_k)

    candidates = []
    for idx in I[0]:
        if idx < len(final_df):
            row = final_df.iloc[idx]
            genre_match = any(g.lower() == genre.lower()
                              for g in row['genres_parsed'])
            rating_match = row['vote_average'] >= float(min_rating)
            actor_match = True
            if actor:
                actor_match = any(actor.lower() in a.lower()
                                  for a in row['actors'])

            if genre_match and rating_match and actor_match:
                candidates.append({
                    'title': row['title'],
                    'rating': row['vote_average'],
                    'genres': row['genres_parsed'],
                    'actors': row['actors'][:5],
                    'overview': row['overview']
                })
                if len(candidates) >= 5:
                    break

    if not candidates:
        gpt_fallback_query = f"""
Recommend a {genre} movie rated above {min_rating}{' starring ' + actor if actor else ''}.
Respond in this exact format using Markdown stars (**):

**Title**: <Movie Title>
**Genres**: <Genre1, Genre2>
**Actors**: <Top 3 actors>
**Rating**: <IMDb rating>
**Why**: <Why this movie is recommended>
**Trailer**: <YouTube trailer URL>
"""

        try:
            response = client.chat.completions.create(
                model="openai/gpt-4.1",
                messages=[
                    {"role": "system", "content": "You're a helpful movie assistant. Return the best movie in the specified format."},
                    {"role": "user", "content": gpt_fallback_query}
                ],
                temperature=0.8,
                top_p=1,
            )
            suggestion = response.choices[0].message.content.strip()
            return [suggestion]
        except Exception as e:
            return [f"🤖 GPT fallback failed: {str(e)}"]

    movie_descriptions = ""
    for i, movie in enumerate(candidates, 1):
        movie_descriptions += (
            f"{i}. Title: {movie['title']}\n"
            f"   Rating: {movie['rating']:.1f}\n"
            f"   Genres: {', '.join(movie['genres'])}\n"
            f"   Stars: {', '.join(movie['actors'])}\n"
            f"   Overview: {movie['overview'][:300]}...\n\n"
        )

    gpt_prompt = f"""
The user is looking for a {genre} movie rated above {min_rating}{' starring ' + actor if actor else ''}.
Here are some candidates:

{movie_descriptions}
Pick the best one and respond in this format using Markdown stars (**):

**Title**: <Movie Title>
**Genres**: <Genre1, Genre2>
**Actors**: <Top 3 actors>
**Rating**: <IMDb rating>
**Why**: <Why this is the best recommendation>
**Trailer**: <YouTube trailer URL>
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4.1",
            messages=[
                {"role": "system", "content": "You're a helpful movie assistant. Pick the best movie from the list and return it in the specified format."},
                {"role": "user", "content": gpt_prompt}
            ],
            temperature=0.7,
            top_p=1,
        )
        suggestion = response.choices[0].message.content.strip()
        return [suggestion]
    except Exception as e:
        return [f"🤖 GPT selection failed: {str(e)}"]
