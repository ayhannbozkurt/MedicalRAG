from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
from typing import List

class BiomedEmbeddings(EmbeddingFunction):
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", token: str = None):
        self.model = SentenceTransformer(model_name, token=token)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist()
