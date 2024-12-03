from pdf_processor_v2 import MedicalPDFProcessor
from embedding_store import DrugEmbeddingStore
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@dataclass
class QueryResult:
    content: str
    source: str
    confidence: float
    metadata: Dict
    retrieved_date: str

class MedicalRAGSystem:
    def __init__(self):
        """Initialize the enhanced Medical RAG system"""
        self.pdf_processor = MedicalPDFProcessor()
        self.drug_store = DrugEmbeddingStore()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Load custom prompts
        self.load_prompts()
    
    def load_prompts(self):
        """Load custom prompts for different query types"""
        self.prompts = {
            "drug_interaction": """Sen bir klinik eczacısın. Verilen ilaç etkileşimi bilgilerini analiz et ve şu formatta yanıtla:

1. Etkileşim Riski:
   - Yüksek risk etkileşimleri
   - Orta risk etkileşimleri
   - Düşük risk etkileşimleri

2. Klinik Önem:
   - Her etkileşimin klinik önemi
   - Hangi hasta gruplarında daha riskli

3. Yönetim Stratejisi:
   - İzlenmesi gereken parametreler
   - Doz ayarlama önerileri
   - Alternatif tedavi seçenekleri

4. Kanıt Düzeyi:
   - Her bilgi için kanıt düzeyi
   - Kaynakların güvenilirliği

Bilgiler:
{context}

Soru: {question}""",
            
            "perioperative": """Sen bir anestezi uzmanısın. Verilen perioperatif ilaç yönetimi bilgilerini analiz et ve şu formatta yanıtla:

1. Preoperatif Değerlendirme:
   - İlacın kesilme zamanı
   - Kesilme kararının gerekçesi
   - Risk değerlendirmesi

2. İntraoperatif Yönetim:
   - Anestezi etkileşimleri
   - İzlenmesi gereken parametreler
   - Acil durumlar için hazırlık

3. Postoperatif Plan:
   - İlaca ne zaman başlanacağı
   - İzlem gereksinimleri
   - Olası komplikasyonlar

4. Özel Durumlar:
   - Hangi durumlarda plan değişebilir
   - Risk faktörleri
   - Alternatif stratejiler

Bilgiler:
{context}

Soru: {question}""",
            
            "general": """Sen bir klinik farmakolog uzmansın. Verilen bilgileri analiz et ve şu formatta yanıtla:

1. Klinik Özet:
   - Ana bulgular
   - Önemli noktalar
   - Klinik uygulamaya yansımaları

2. Detaylı Analiz:
   - Mekanizma açıklaması
   - Farmakokinetik/Farmakodinamik özellikler
   - Hasta özellikleri

3. Pratik Öneriler:
   - Klinik karar noktaları
   - İzlem gereksinimleri
   - Hasta eğitimi

4. Kanıt Değerlendirmesi:
   - Kanıt düzeyleri
   - Kısıtlılıklar
   - Araştırma gereksinimleri

Bilgiler:
{context}

Soru: {question}"""
        }
    
    def determine_query_type(self, question: str) -> str:
        """Determine the type of medical query"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["interaction", "combine", "together", "mix"]):
            return "drug_interaction"
        elif any(word in question_lower for word in ["surgery", "operation", "preoperative", "postoperative"]):
            return "perioperative"
        else:
            return "general"
    
    def get_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from LLM with error handling and retry logic"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for medical accuracy
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent medical advice
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in LLM response: {e}")
            return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."
    
    def query(self, question: str, n_results: int = 5) -> str:
        """
        Enhanced query processing with multiple sources and structured response
        """
        try:
            # Determine query type
            query_type = self.determine_query_type(question)
            logger.info(f"Query type determined as: {query_type}")
            
            # Get results from both sources
            pdf_results = self.pdf_processor.query(
                query=question,
                n_results=n_results,
                has_interactions=(query_type == "drug_interaction")
            )
            
            drug_results = self.drug_store.collection.query(
                query_texts=[question],
                n_results=n_results
            )
            
            # Combine and format results
            context = ""
            
            # Add drug database information
            if drug_results['documents'][0]:
                context += "\nİlaç Veritabanı Bilgileri:\n"
                for doc, metadata in zip(drug_results['documents'][0], drug_results['metadatas'][0]):
                    context += f"\nKaynak: {metadata.get('source', 'İlaç Veritabanı')}\n{doc}\n"
            
            # Add academic research information
            if pdf_results:
                context += "\nAkademik Araştırma Bilgileri:\n"
                for result in pdf_results:
                    context += f"\nKaynak: {result['source']}\n"
                    context += f"İşlenme Tarihi: {result['processed_date']}\n"
                    context += f"{result['content']}\n"
            
            # If no results found
            if not context.strip():
                return "Bu soru için uygun bilgi bulunamadı. Lütfen soruyu daha spesifik hale getirin veya başka bir şekilde sorun."
            
            # Get appropriate prompt template
            prompt_template = self.prompts[query_type]
            
            # Generate response
            response = self.get_llm_response(
                system_prompt="""Sen bir klinik farmakolog uzmansın. Verilen bilgileri kullanarak soruları detaylı ve doğru bir şekilde yanıtla.
                Yanıtlarında hem ilaç veritabanından hem de akademik araştırmalardan gelen bilgileri kullan.
                Her zaman kanıta dayalı tıp prensiplerini gözet ve bilgilerin kaynağını belirt.""",
                user_prompt=prompt_template.format(context=context, question=question)
            )
            
            # Log query for future analysis
            self._log_query(question, query_type, context, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."
    
    def process_query(self, query: str) -> QueryResult:
        """Process a medical query and return a structured response"""
        try:
            # Get relevant context from the vector store
            results = self.drug_store.search(query, k=3)
            context = "\n".join([r['page_content'] for r in results])
            
            # Determine query type
            query_type = "general"
            if any(word in query.lower() for word in ["etkileşim", "interaction", "birlikte"]):
                query_type = "drug_interaction"
            elif any(word in query.lower() for word in ["ameliyat", "operasyon", "surgery"]):
                query_type = "perioperative"
            
            # Get the appropriate prompt template
            prompt = self.prompts.get(query_type, self.prompts["general"])
            
            # Format the prompt with context and query
            formatted_prompt = prompt.format(context=context, question=query)
            
            # Generate response using OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Sen bir klinik eczacısın."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3
            )
            
            # Extract the response content
            content = response.choices[0].message.content
            
            # Create and return QueryResult
            return QueryResult(
                content=content,
                source=", ".join([r.metadata.get("source", "Unknown") for r in results]),
                confidence=float(response.choices[0].finish_reason == "stop"),
                metadata={"query_type": query_type},
                retrieved_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def _log_query(self, question: str, query_type: str, context: str, response: str):
        """Log query details for analysis and improvement"""
        try:
            log_dir = "query_logs"
            os.makedirs(log_dir, exist_ok=True)
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "query_type": query_type,
                "context_length": len(context),
                "response_length": len(response),
                "response": response
            }
            
            log_file = os.path.join(log_dir, f"query_log_{datetime.now().strftime('%Y%m')}.json")
            
            # Append to existing log or create new
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except FileNotFoundError:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error logging query: {e}")

def main():
    # Test the enhanced system
    rag = MedicalRAGSystem()
    
    # Test queries
    queries = [
        "My patient uses turmeric for a long time. Which drugs can turmeric interact with during perioperative management? When should my patient stop turmeric before the surgery?",
        "What are the potential interactions between metformin and ACE inhibitors in diabetic patients with kidney disease?",
        "How should I manage anticoagulation in a patient taking warfarin who needs emergency surgery?"
    ]
    
    for query in queries:
        print("\nSoru:", query)
        print("\nYanıt:", rag.query(query))
        print("\n" + "="*80)

if __name__ == "__main__":
    main()
