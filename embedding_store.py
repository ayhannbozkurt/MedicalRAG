from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.api.types import EmbeddingFunction
import chromadb
import json
import logging
from tqdm import tqdm
from typing import List, Dict
import os
from dotenv import load_dotenv
from embeddings import BiomedEmbeddings

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class DrugEmbeddingStore:
    def __init__(self, json_dir="data/json", force_reload=False):
        """
        Initialize the drug embedding store
        
        Args:
            json_dir: Directory containing JSON files with drug data
            force_reload: If True, force reprocessing of JSON files
        """
        # Initialize embeddings
        self.embeddings = BiomedEmbeddings(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            token=HF_TOKEN
        )
        
        # Initialize ChromaDB client
        self.client = PersistentClient(path="db")
        self.collection = self.client.get_or_create_collection(
            name="drug_interactions",
            embedding_function=self.embeddings
        )
        
        # Store JSON directory
        self.json_dir = json_dir
        
        # Process JSON files only if collection is empty or force_reload is True
        if force_reload or self.collection.count() == 0:
            logger.info("Processing JSON files...")
            self.process_json_files()
        else:
            logger.info(f"Using existing collection with {self.collection.count()} items")
    
    def process_json_files(self, batch_size: int = 32):
        """Process all JSON files in the json_dir"""
        if not os.path.exists(self.json_dir):
            logger.warning(f"JSON directory {self.json_dir} does not exist")
            return
        
        for filename in os.listdir(self.json_dir):
            if filename.endswith('.json'):
                json_path = os.path.join(self.json_dir, filename)
                try:
                    self.add_drugs(json_path, batch_size)
                    logger.info(f"Successfully processed {filename}")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
    
    def _format_drug_text(self, drug: Dict, interactions: List[Dict]) -> str:
        """Format drug and interaction information into text"""
        text = f"""
        Drug: {drug['label']}
        Type: {drug['type']}
        Description: {drug['properties']['description']}
        
        Interactions:
        {self._format_interactions(interactions)}
        """
        return text.strip()
    
    def _format_interactions(self, interactions: List[Dict]) -> str:
        """Format interaction information into text"""
        return "\n".join([
            f"- Interacts with {interaction['target']}: {interaction['properties']['description']}"
            for interaction in interactions
        ])
    
    def add_drugs(self, json_file: str, batch_size: int = 32):
        """Load drug data from a JSON file and create embeddings"""
        try:
            logger.info(f"Loading data from {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process drugs in batches
            drugs = data['nodes']
            total_batches = (len(drugs) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(drugs), batch_size), desc=f"Processing {os.path.basename(json_file)}"):
                batch = drugs[i:i + batch_size]
                
                documents = []
                metadatas = []
                ids = []
                
                for drug in batch:
                    # Find drug interactions
                    interactions = [
                        edge for edge in data['edges']
                        if edge['source'] == drug['id']
                    ]
                    
                    # Create text and add to list
                    doc_text = self._format_drug_text(drug, interactions)
                    documents.append(doc_text)
                    
                    # Add metadata and ID
                    metadatas.append({
                        "drugbank_id": drug['id'],
                        "name": drug['label'],
                        "type": drug['type'],
                        "interaction_count": len(interactions),
                        "source_file": os.path.basename(json_file)
                    })
                    ids.append(f"{drug['id']}_{os.path.basename(json_file)}")
                
                # Add batch to database
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Successfully added {len(drugs)} drugs from {json_file}")
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            raise
    
    def query_similar_drugs(self, query: str, n_results: int = 5, 
                          min_interactions: int = 0) -> Dict:
        """
        Query similar drugs
        
        Args:
            query: Query text
            n_results: Number of results to return
            min_interactions: Minimum number of interactions filter
        """
        where = {}
        if min_interactions > 0:
            where = {"interaction_count": {"$gte": min_interactions}}
            
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        return self._format_results(results)
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format query results"""
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'drug_id': results['ids'][0][i],
                'name': results['metadatas'][0][i]['name'],
                'type': results['metadatas'][0][i]['type'],
                'interaction_count': results['metadatas'][0][i]['interaction_count'],
                'text': results['documents'][0][i]
            })
        return formatted
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """
        Search for documents matching the query
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of documents with their metadata
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            # Format results
            documents = []
            for i in range(len(results['documents'][0])):
                documents.append({
                    'page_content': results['documents'][0][i],
                    'metadata': {
                        'source': results['metadatas'][0][i].get('name', 'Unknown'),
                        'score': results['distances'][0][i] if 'distances' in results else None
                    }
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

def main():
    # Test the embedding store
    store = DrugEmbeddingStore()
    
    # Test query
    query = "What drugs interact with aspirin?"
    print("\nQuery:", query)
    
    results = store.collection.query(
        query_texts=[query],
        n_results=3
    )
    
    print("\nResults:")
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"\nDrug: {meta['name']} (Source: {meta['source_file']})")
        print(doc)

if __name__ == "__main__":
    main()
