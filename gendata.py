import json
import random
import uuid

ITA_POS = [
    "Ho acquistato un iPhone 15 presso Apple Roma Est e la consegna DHL è stata rapidissima.",
    "Ottima esperienza con Amazon Italia: il mio Lenovo ThinkPad è arrivato in perfette condizioni a Milano.",
    "Il servizio clienti di Booking.com mi ha aiutato subito durante il soggiorno a Firenze.",
    "Ho ordinato delle cuffie Sony WH-1000XM5 su MediaWorld e il ritiro a Torino è stato perfetto.",
    "L'assistenza di PayPal Italia ha risolto il problema del pagamento in meno di un'ora.",
    "Il Samsung Galaxy S24 comprato da Unieuro a Napoli funziona benissimo.",
]

ITA_NEG = [
    "Il servizio clienti di TIM non ha risolto il guasto sulla mia linea a Roma.",
    "Il software di Acme Corp va in crash su Windows 11 ogni volta che apro il report mensile.",
    "Ho comprato un MacBook Air da Apple Milano, ma il supporto Apple non ha gestito il difetto dello schermo.",
    "La consegna DHL del mio ordine Amazon è arrivata a Bologna con tre giorni di ritardo e il pacco era danneggiato.",
    "L'app di Intesa Sanpaolo si blocca continuamente sul mio iPhone 14 dopo l'ultimo aggiornamento.",
    "Booking.com non ha gestito correttamente il reclamo relativo all'hotel di Venezia.",
]

ENG_POS = [
    "Amazon delivered my Sony WH-1000XM5 to London in one day, and the packaging was excellent.",
    "I had a great experience with Booking.com during my trip to Barcelona.",
    "The Apple Store in Berlin replaced my iPhone 15 battery quickly and professionally.",
    "PayPal support resolved my refund issue for an eBay order within a few hours.",
    "My Lenovo ThinkPad from MediaMarkt in Munich works exactly as described.",
    "DHL delivered my Samsung Galaxy S24 to Paris earlier than expected.",
]

ENG_NEG = [
    "The support team at PayPal ignored my refund request for an order from eBay.",
    "My Dell XPS 13 purchased from MediaMarkt in Berlin overheats after the latest update.",
    "Booking.com did not respond to my complaint about the hotel in Milan.",
    "The new Acme Corp software crashes every time I upload files on Windows 11.",
    "Apple Support failed to solve the screen issue on my MacBook Pro in Rome.",
    "DHL lost my Amazon package during delivery to Madrid.",
]

SOURCES = ["source-web", "source-app", "source-email"]


def generate_dataset(
    num_records=20,
    output_file="test_data.json",
    negative_ratio=0.4,
):
    data = []

    pools = [
        ("it", ITA_POS, ITA_NEG),
        ("en", ENG_POS, ENG_NEG),
    ]

    for _ in range(num_records):
        lang, pos_pool, neg_pool = random.choice(pools)
        is_negative = random.random() < negative_ratio
        text = random.choice(neg_pool if is_negative else pos_pool)

        data.append(
            {
                "ID": str(uuid.uuid4()),
                "SourceID": random.choice(SOURCES),
                "Text": text,
                "Lang": lang,
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    generate_dataset()
