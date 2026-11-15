from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from config import settings

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"   
DIMENSION = 768
INDEX_NAME = "medical-bot"

model = SentenceTransformer(MODEL_NAME)
# pc = Pinecone(api_key="PINECONE_API_KEY")




@lru_cache()
def clients():
    if not settings.pinecone_api_key:
        raise ValueError("Pinecone API key must be set in the environment variables.")
    return Pinecone(api_key=settings.pinecone_api_key)




def ensure_index():
    pc = clients()
    pc.delete_index(settings.pinecone_index)
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(name=settings.pinecone_index, dimension=DIMENSION, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    return pc.Index(settings.pinecone_index)

def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, normalize_embeddings=True).tolist()


def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query])[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{'page': m.metadata.get('page'), 'text': m.metadata.get('text',''), 'score': round(m.score,3), 'source': 'Gale Encyclopedia (2002)'} for m in result.matches if m.score >= 0.30]