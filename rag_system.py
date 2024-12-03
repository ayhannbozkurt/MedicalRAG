from embedding_store import DrugEmbeddingStore
from pdf_processor import MedicalPDFProcessor
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
import logging
import requests

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class DrugRAGSystem:
    def __init__(self):
        """Initialize the RAG system."""
        load_dotenv()
        
        # Download data files if they don't exist
        self.ensure_data_files()
        
        # İki farklı veri kaynağı için store'ları başlat
        self.drug_store = DrugEmbeddingStore(force_reload=False)  # Varsayılan olarak mevcut veritabanını kullan
        self.pdf_store = MedicalPDFProcessor(force_reload=False)  # Varsayılan olarak mevcut veritabanını kullan
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        
        # Prompt template oluştur
        self.prompt = PromptTemplate(
            input_variables=["question", "drug_context", "research_context"],
            template="""Bir ilaç uzmanı olarak, aşağıdaki bilgileri kullanarak soruyu yanıtla.

İlaç Veritabanı Bilgileri:
{drug_context}

Akademik Araştırma Bilgileri:
{research_context}

Soru: {question}

Yanıt: Verilen bilgileri birleştirerek kapsamlı bir yanıt oluşturacağım:"""
        )
        
        # Create chain
        self.chain = (
            {
                "drug_context": lambda x: x["drug_context"],
                "research_context": lambda x: x["research_context"],
                "question": lambda x: x["question"]
            }
            | self.prompt 
            | self.llm
            | StrOutputParser()
        )
    
    def ensure_data_files(self):
        """Ensure all required data files are present."""
        # Create data directories if they don't exist
        os.makedirs('data/json', exist_ok=True)
        os.makedirs('data/pdfs', exist_ok=True)
        
        # Check and download JSON file
        json_path = 'data/json/drug_graph.json'
        if not os.path.exists(json_path):
            json_url = os.getenv('DRUG_JSON_URL')
            if json_url:
                print("Downloading drug database...")
                self.download_file(json_url, json_path)
            else:
                raise ValueError("DRUG_JSON_URL environment variable is not set")
        
        # Check and download PDF files
        pdf_urls = os.getenv('PDF_URLS', '').split(',')
        for url in pdf_urls:
            if url.strip():
                filename = url.split('/')[-1]
                pdf_path = f'data/pdfs/{filename}'
                if not os.path.exists(pdf_path):
                    print(f"Downloading {filename}...")
                    self.download_file(url, pdf_path)
    
    def download_file(self, url: str, path: str):
        """Download a file from URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            raise
    
    def query(self, question: str, n_results: int = 3) -> str:
        """
        Kullanıcı sorusunu hem ilaç veritabanı hem de akademik araştırmalardan yanıtlar
        
        Args:
            question: Kullanıcı sorusu
            n_results: Her kaynaktan kaç sonuç alınacak
        """
        try:
            # İlaç veritabanından bilgi al
            drug_results = self.drug_store.collection.query(
                query_texts=[question],
                n_results=n_results
            )
            
            # PDF'lerden bilgi al
            pdf_results = self.pdf_store.collection.query(
                query_texts=[question],
                n_results=n_results,
                include=["documents", "metadatas"]
            )
            
            # Bağlamları oluştur
            drug_context = "Bilgi bulunamadı."
            if drug_results['documents'][0]:
                drug_context = "\n\n".join(drug_results['documents'][0])
            
            research_context = "Bilgi bulunamadı."
            if pdf_results['documents'][0]:
                research_texts = []
                for doc, meta in zip(pdf_results['documents'][0], pdf_results['metadatas'][0]):
                    research_texts.append(f"Kaynak: {meta['source']}\n{doc}")
                research_context = "\n\n".join(research_texts)
            
            # LLM ile yanıt üret
            response = self.chain.invoke({
                "drug_context": drug_context,
                "research_context": research_context,
                "question": question
            })
            
            # Kaynakları ve alıntıları ekle
            sources = "\n\n=== Kullanılan Kaynaklar ve İlgili Bölümler ===\n"
            
            # İlaç veritabanından alınan bilgiler
            if drug_results['documents'][0]:
                sources += "\nİlaç Veritabanından Alınan Bilgiler:\n"
                sources += "--------------------------------\n"
                for doc in drug_results['documents'][0]:
                    sources += f"{doc}\n\n"
            
            # PDF'lerden alınan bilgiler
            if pdf_results['documents'][0]:
                sources += "\nAkademik Araştırmalardan Alınan Bilgiler:\n"
                sources += "--------------------------------\n"
                for doc, meta in zip(pdf_results['documents'][0], pdf_results['metadatas'][0]):
                    sources += f"Kaynak: {os.path.basename(meta['source'])}\n"
                    sources += f"İlgili Bölüm:\n{doc}\n\n"
            
            return response + sources
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."
    
    def get_drug_interactions(self, drug_name: str) -> str:
        """
        Belirli bir ilacın etkileşimlerini hem veritabanından hem de akademik araştırmalardan sorgular
        
        Args:
            drug_name: İlaç adı
        """
        try:
            # Veritabanından ilaç bilgisi al
            drug_results = self.drug_store.collection.query(
                query_texts=[drug_name],
                n_results=1
            )
            
            # PDF'lerden ilaç bilgisi al
            pdf_results = self.pdf_store.collection.query(
                query_texts=[f"{drug_name} etkileşimleri ve yan etkileri"],
                n_results=3,
                include=["documents", "metadatas"]
            )
            
            # Bağlamları oluştur
            drug_context = "Veritabanında ilaç bilgisi bulunamadı."
            if drug_results['documents'][0]:
                drug_context = drug_results['documents'][0][0]
            
            research_context = "Akademik araştırmalarda bilgi bulunamadı."
            if pdf_results['documents'][0]:
                research_texts = []
                for doc, meta in zip(pdf_results['documents'][0], pdf_results['metadatas'][0]):
                    research_texts.append(f"Kaynak: {meta['source']}\n{doc}")
                research_context = "\n\n".join(research_texts)
            
            # Soru oluştur
            question = f"{drug_name} ilacının diğer ilaçlarla etkileşimlerini ve yan etkilerini detaylı açıkla."
            
            # LLM ile yanıt üret
            response = self.chain.invoke({
                "drug_context": drug_context,
                "research_context": research_context,
                "question": question
            })
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Error getting drug interactions: {e}")
            return f"'{drug_name}' ilacı hakkında bilgi alınırken bir hata oluştu."

def main():
    # RAG sistemini başlat
    rag = DrugRAGSystem()
    
    # İlk kullanımda veya veritabanı boşsa PDF'leri işle
    pdf_dir = "data/pdfs"
    while True:
        try:
            # Kullanıcıdan soru al
            question = input("\nSorunuzu girin (çıkmak için 'q'): ")
            
            if question.lower() == 'q':
                break
            
            # Yanıt al
            print("\nYanıt aranıyor...")
            response = rag.query(question)
            print(f"\nYanıt:\n{response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nHata oluştu: {e}")

if __name__ == "__main__":
    main()
