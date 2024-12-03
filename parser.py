import logging
from typing import Dict, List, Generator
import json
from pathlib import Path
from lxml import etree
from tqdm import tqdm

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DrugBankParser:
    def __init__(self, xml_file: str):
        self.xml_file = Path(xml_file)
        self.ns = {'db': 'http://www.drugbank.ca'}
        
    def get_file_size(self) -> int:
        """XML dosyasının boyutunu döndürür"""
        return self.xml_file.stat().st_size
    
    def clear_element(self, elem):
        """Belleği temizler"""
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    
    def extract_text_safe(self, element, xpath: str, default=None) -> str:
        """Güvenli bir şekilde XML elementinden text çıkarır"""
        try:
            result = element.find(xpath, self.ns)
            return result.text if result is not None else default
        except Exception:
            return default

    def parse_drug_basic_info(self, drug_element) -> Dict:
        """İlacın temel bilgilerini parse eder"""
        try:
            return {
                'drugbank_id': self.extract_text_safe(drug_element, 'db:drugbank-id[@primary="true"]'),
                'name': self.extract_text_safe(drug_element, 'db:name'),
                'description': self.extract_text_safe(drug_element, 'db:description'),
                'state': self.extract_text_safe(drug_element, 'db:state'),
                'type': drug_element.get('type')
            }
        except Exception as e:
            logger.error(f"Error parsing basic info: {e}")
            return {}

    def parse_drug_interactions(self, drug_element) -> List[Dict]:
        """İlaç etkileşimlerini parse eder"""
        interactions = []
        try:
            for interaction in drug_element.findall('.//db:drug-interaction', self.ns):
                interaction_data = {
                    'drugbank_id': self.extract_text_safe(interaction, 'db:drugbank-id'),
                    'name': self.extract_text_safe(interaction, 'db:name'),
                    'description': self.extract_text_safe(interaction, 'db:description')
                }
                if interaction_data['drugbank_id']:  # Sadece geçerli ID'leri ekle
                    interactions.append(interaction_data)
        except Exception as e:
            logger.error(f"Error parsing interactions: {e}")
        return interactions

    def drug_generator(self) -> Generator:
        """İlaçları yield eden generator"""
        context = etree.iterparse(self.xml_file, events=('end',), tag=f'{{{self.ns["db"]}}}drug')
        for event, elem in context:
            yield elem
            self.clear_element(elem)

    def process_in_batches(self, batch_size: int = 100) -> Dict:
        """İlaçları batch'ler halinde işler"""
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        total_size = self.get_file_size()
        batch = []
        
        with tqdm(total=total_size, desc="Processing drugs", unit='B', unit_scale=True) as pbar:
            for drug_elem in self.drug_generator():
                try:
                    # İlaç bilgilerini parse et
                    basic_info = self.parse_drug_basic_info(drug_elem)
                    if basic_info and basic_info.get('drugbank_id'):
                        node = {
                            'id': basic_info['drugbank_id'],
                            'label': basic_info['name'],
                            'type': basic_info['type'],
                            'properties': basic_info
                        }
                        graph_data['nodes'].append(node)
                        
                        # Etkileşimleri parse et
                        interactions = self.parse_drug_interactions(drug_elem)
                        for interaction in interactions:
                            if interaction.get('drugbank_id'):
                                edge = {
                                    'source': basic_info['drugbank_id'],
                                    'target': interaction['drugbank_id'],
                                    'label': 'interacts_with',
                                    'properties': {
                                        'description': interaction.get('description')
                                    }
                                }
                                graph_data['edges'].append(edge)
                        
                        batch.append(node)
                        if len(batch) >= batch_size:
                            batch = []
                            
                except Exception as e:
                    logger.error(f"Error processing drug element: {e}")
                
                pbar.update(len(str(drug_elem)))
        
        return graph_data

    def save_to_json(self, output_file: str, batch_size: int = 100):
        """Graph verisini JSON olarak kaydeder"""
        try:
            logger.info("Starting to parse XML...")
            graph_data = self.process_in_batches(batch_size)
            
            logger.info(f"Parsing complete. Found {len(graph_data['nodes'])} drugs and {len(graph_data['edges'])} interactions")
            
            logger.info(f"Saving to {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Save complete!")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise

def main():
    try:
        input_file = 'drugbank_fixed.xml'
        output_file = 'drug_graph.json'
        
        if not Path(input_file).exists():
            logger.error(f"Input file {input_file} does not exist!")
            return
        
        parser = DrugBankParser(input_file)
        parser.save_to_json(output_file)
        
    except Exception as e:
        logger.error(f"Main process error: {e}")
        raise

if __name__ == "__main__":
    main()