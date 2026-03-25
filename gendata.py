import json
import uuid
import random

ITA_POS = ["Ottimo prodotto, spedizione velocissima.", "La qualità è eccellente e l'interfaccia utente è molto intuitiva."]
ITA_NEG = ["Pessima esperienza, il prodotto è arrivato rotto.", "Il sistema va in crash continuamente. Inaccettabile."]
ENG_POS = ["Highly recommend this service. It works exactly as described.", "Fast delivery. The build quality is fantastic."]
ENG_NEG = ["Terrible customer service. I want a refund immediately.", "The new update broke everything. It's completely unusable."]

def generate_dataset(num_records=10, output_file="test_data.json"):
    data = []
    sources = ["source-web", "source-app", "source-email"]
    
    for _ in range(num_records):
        lang_pool = random.choice([(ITA_POS, ITA_NEG), (ENG_POS, ENG_NEG)])
        is_negative = random.random() < 0.4 # 40% di probabilità di feedback negativo
        text = random.choice(lang_pool[1] if is_negative else lang_pool[0])

        data.append({
            "ID": str(uuid.uuid4()),
            "SourceID": random.choice(sources),
            "Text": text
        })
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    generate_dataset()
