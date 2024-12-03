from typing import List, Dict, Optional
import os
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import PersistentClient
import chromadb
from dotenv import load_dotenv
import logging
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
import spacy
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from embeddings import BiomedEmbeddings
import json

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

@dataclass
class ExtractedInfo:
    drug_name: str
    drug_class: Optional[str]
    interactions: List[Dict]
    contraindications: List[str]
    dosage: Optional[Dict]
    side_effects: List[str]
    evidence_level: str
    source: str
    publication_date: Optional[str]

class BiomedicalNLP:
    def __init__(self):
        """Initialize NLP components"""
        # Load SpaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Custom patterns for drug information
        self.matcher = self.nlp.add_pipe("entity_ruler")
        self.matcher.add_patterns([
            {"label": "DRUG", "pattern": [{"LOWER": {"REGEX": "(tablet|capsule|injection|solution)"}}]},
            {"label": "DOSAGE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["mg", "ml", "g", "mcg"]}}]},
            {"label": "FREQUENCY", "pattern": [{"LOWER": {"REGEX": "(daily|weekly|monthly|twice|once)"}}]},
        ])
    
    def extract_drug_info(self, text: str) -> ExtractedInfo:
        """Extract structured information about drugs from text"""
        doc = self.nlp(text)
        
        # Initialize extraction results
        info = ExtractedInfo(
            drug_name="",
            drug_class=None,
            interactions=[],
            contraindications=[],
            dosage={},
            side_effects=[],
            evidence_level="",
            source="",
            publication_date=None
        )
        
        # Extract drug names and classes
        for ent in doc.ents:
            if ent.label_ == "CHEMICAL":
                info.drug_name = ent.text
            elif ent.label_ == "DRUG":
                info.drug_class = ent.text
        
        # Extract interactions and contraindications
        for sent in doc.sents:
            if any(word in sent.text.lower() for word in ["interact", "interaction", "contraindicated"]):
                if "interact" in sent.text.lower():
                    info.interactions.append({"description": sent.text})
                else:
                    info.contraindications.append(sent.text)
        
        # Extract dosage information
        dosage_ents = [ent for ent in doc.ents if ent.label_ in ["DOSAGE", "FREQUENCY"]]
        if dosage_ents:
            info.dosage = {
                "amount": next((ent.text for ent in dosage_ents if ent.label_ == "DOSAGE"), None),
                "frequency": next((ent.text for ent in dosage_ents if ent.label_ == "FREQUENCY"), None)
            }
        
        # Extract side effects
        for sent in doc.sents:
            if any(word in sent.text.lower() for word in ["side effect", "adverse", "toxicity"]):
                info.side_effects.append(sent.text)
        
        return info

class MedicalPDFProcessor:
    def __init__(self, persist_directory="db", force_reload=False):
        """
        Enhanced PDF processing and vector store management.
        
        Args:
            persist_directory: Directory for vector store persistence
            force_reload: If True, force reprocessing of all PDFs
        """
        # Initialize components
        self.embeddings = BiomedEmbeddings(token=HF_TOKEN)
        self.nlp = BiomedicalNLP()
        
        # Initialize ChromaDB client
        self.client = PersistentClient(path=persist_directory)
        
        try:
            self.collection = self.client.get_collection(
                name="medical_documents",
                embedding_function=self.embeddings
            )
            if force_reload:
                logger.info("Force reload requested. Deleting existing collection...")
                self.client.delete_collection(name="medical_documents")
                self.collection = self.client.create_collection(
                    name="medical_documents",
                    embedding_function=self.embeddings
                )
        except:
            self.collection = self.client.create_collection(
                name="medical_documents",
                embedding_function=self.embeddings
            )
        
        # Configure text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Track processed documents
        self.processed_docs = defaultdict(lambda: {
            "last_processed": None,
            "chunks": 0,
            "extracted_info": None
        })
        
        # Load processing history if exists
        self._load_processing_history()
    
    def _load_processing_history(self):
        """Load document processing history from disk"""
        history_file = os.path.join("db", "pdf_processing_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    for doc_path, info in history.items():
                        info["last_processed"] = datetime.fromisoformat(info["last_processed"]) if info["last_processed"] else None
                        self.processed_docs[doc_path].update(info)
            except Exception as e:
                logger.warning(f"Could not load processing history: {e}")

    def _save_processing_history(self):
        """Save document processing history to disk"""
        history_file = os.path.join("db", "pdf_processing_history.json")
        try:
            history = {}
            for doc_path, info in self.processed_docs.items():
                history[doc_path] = {
                    "last_processed": info["last_processed"].isoformat() if info["last_processed"] else None,
                    "chunks": info["chunks"],
                    "extracted_info": info["extracted_info"]
                }
            with open(history_file, 'w') as f:
                json.dump(history, f)
        except Exception as e:
            logger.warning(f"Could not save processing history: {e}")
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a medical PDF and extract structured information.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            Dict containing extracted information and processing stats
        """
        try:
            # Check if already processed recently
            if pdf_path in self.processed_docs:
                last_processed = self.processed_docs[pdf_path]["last_processed"]
                if last_processed and (datetime.now() - last_processed).days < 7:
                    logger.info(f"Using cached results for {pdf_path}")
                    return self.processed_docs[pdf_path]
            
            # Extract text from PDF
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page in doc:
                # Extract text blocks with their positions
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        text = " ".join(span["text"] for line in block["lines"] 
                                      for span in line["spans"])
                        if len(text.strip()) > 50:  # Filter out short blocks
                            text_blocks.append(text)
            
            # Clean and process text
            processed_text = "\n\n".join(text_blocks)
            processed_text = clean_text(processed_text)
            
            # Extract structured information
            extracted_info = self.nlp.extract_drug_info(processed_text)
            extracted_info.source = pdf_path
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(processed_text)
            
            # Prepare documents for vector store
            documents = []
            metadatas = []
            ids = []
            
            base_name = os.path.basename(pdf_path)
            
            for i, chunk in enumerate(chunks):
                # Extract key information from chunk
                chunk_info = self.nlp.extract_drug_info(chunk)
                
                # Create rich metadata
                metadata = {
                    "source": pdf_path,
                    "chunk_id": i,
                    "drug_mentions": chunk_info.drug_name,
                    "has_interactions": bool(chunk_info.interactions),
                    "has_dosage": bool(chunk_info.dosage),
                    "has_side_effects": bool(chunk_info.side_effects),
                    "processed_date": datetime.now().isoformat()
                }
                
                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(f"{base_name}_chunk_{i}")
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update processed docs tracking
            self.processed_docs[pdf_path] = {
                "last_processed": datetime.now(),
                "chunks": len(chunks),
                "extracted_info": extracted_info
            }
            
            doc.close()
            logger.info(f"Successfully processed {base_name} into {len(documents)} chunks")
            return self.processed_docs[pdf_path]
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            raise
    
    def process_directory(self, directory_path: str):
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Directory containing PDF files
            
        Returns:
            Dict[str, Dict]: Processing results for each file
        """
        results = {}
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            logger.warning(f"Directory {directory_path} does not exist")
            return results
        
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return results
        
        # Check if collection is empty
        if self.collection.count() == 0:
            logger.info("Collection is empty. Processing all PDFs...")
            force_process = True
        else:
            force_process = False
        
        # Process each PDF file
        for filename in pdf_files:
            pdf_path = os.path.join(directory_path, filename)
            
            # Skip if already processed and not forcing
            if not force_process and pdf_path in self.processed_docs:
                last_processed = self.processed_docs[pdf_path]["last_processed"]
                if last_processed and (datetime.now() - last_processed).days < 7:
                    logger.info(f"Skipping {filename} - processed within last 7 days")
                    results[filename] = self.processed_docs[pdf_path]
                    continue
            
            try:
                result = self.process_pdf(pdf_path)
                results[filename] = result
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                results[filename] = {"error": str(e)}
        
        # Save processing history
        self._save_processing_history()
        
        return results
    
    def query(self, query_text: str, n_results: int = 5, filter_metadata: Dict = None) -> Dict:
        """
        Query the vector store with advanced filtering.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            Dict containing query results and metadata
        """
        try:
            query_params = {
                "query_texts": [query_text],
                "n_results": n_results,
                "include": ["metadatas", "distances", "documents"]
            }
            
            if filter_metadata:
                query_params["where"] = filter_metadata
            
            results = self.collection.query(**query_params)
            
            # Process and structure results
            processed_results = []
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                processed_results.append({
                    "text": doc,
                    "metadata": meta,
                    "relevance_score": 1 - (distance / 2)  # Convert distance to similarity score
                })
            
            return {
                "results": processed_results,
                "query": query_text,
                "total_results": len(processed_results)
            }
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise

def clean_text(text: str) -> str:
    """
    PDF'den çıkarılan metni temizler ve düzenler
    """
    # Gereksiz boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    # Satır sonlarını düzelt
    text = text.replace('- ', '')
    
    # Noktalama işaretlerinden sonra boşluk ekle
    text = re.sub(r'([.,!?:;])([^\s])', r'\1 \2', text)
    
    # Parantezlerin içinde ve dışında boşluk bırak
    text = re.sub(r'\(\s*', ' (', text)
    text = re.sub(r'\s*\)', ') ', text)
    
    # URL'leri koru
    text = re.sub(r'(https?://\S+)\s+', r'\1', text)
    
    return text.strip()

def main():
    # Test fonksiyonu
    processor = MedicalPDFProcessor()
    
    # Örnek kullanım
    results = processor.process_directory("/Users/ayhanbozkurt/Documents/Medical-RAG")
    print("\nİşlenen dosyalar ve chunk sayıları:")
    for filename, result in results.items():
        print(f"{filename}: {result['chunks']} chunks")
    
    # Örnek sorgu
    print("\nSorgu testi:")
    query_results = processor.query("İlaç etkileşimleri nelerdir?")
    
    # Sonuçları göster
    for i, result in enumerate(query_results["results"]):
        print(f"\nSonuç {i+1} (Benzerlik: {result['relevance_score']:.2f}):")
        print(f"Kaynak: {result['metadata']['source']}")
        print(f"Metin: {result['text'][:200]}...")

if __name__ == "__main__":
    main()