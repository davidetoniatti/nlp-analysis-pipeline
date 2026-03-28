# Scalable NLP Analysis Pipeline

Il sistema acquisisce documenti, li invia a un servizio NLP e salva in Postgres le esecuzioni e i risultati finali.

## Scopo del progetto

Per ogni documento in input, il sistema:
- raccoglie il testo,
- esegue sentiment analysis,
- estrae entità nominate,
- genera un riassunto solo per i testi negativi,
- salva in modo persistente run e risultati nel database.

## Architettura

Il sistema è diviso in tre componenti principali:

- **Postgres**: salva documenti, analysis run, risultati, metadati della sorgente e versioni dei modelli.
- **Backend Go / orchestratore**: consuma messaggi, costruisce batch, chiama il servizio NLP, gestisce i retry e persiste i risultati verificati.
- **Servizio NLP in Python**: espone un'API HTTP che esegue sentiment, NER e summarization condizionale.

## Avvio rapido

### Requisiti

- Docker
- Docker Compose

### Avvio

```bash
docker compose up --build
```

### Configurazione

Variabili di ambiente:
- `DATABASE_URL` per il backend Go
- `INFERENCE_URL` per il backend Go
- `GROQ_API_KEY` per la summarization via LLM
- `REQUEST_TIMEOUT_S`
- `LLM_TIMEOUT_S`
- `MAX_TEXT_CHARS`

## Flusso dei dati

Il flusso della pipeline è il seguente:

1. Il backend Go carica i documenti mock da `test_data.json`.
2. I documenti vengono raggruppati in batch.
3. Il backend chiama l'endpoint `/analyze` del servizio Python.
4. Il servizio NLP esegue:
   - sentiment analysis multilingua,
   - NER multilingua,
   - summarization solo per i testi negativi.
5. Il backend riceve i risultati del batch.
6. I risultati vengono persistiti in Postgres dentro una transazione.
7. La persistenza viene verificata prima dell'ack finale del batch.

## Schema del database

Il database tiene traccia sia dei dati applicativi sia dei metadati di elaborazione.

Tabelle principali:

- `sourcemetadata`: origine del documento
- `document`: documento grezzo in ingresso
- `modelversion`: registry minimale dei modelli usati
- `analysisrun`: una singola esecuzione di analisi su un documento
- `analysisresult`: output finale strutturato di una run completata con successo

In sintesi:
- una sorgente può produrre molti documenti,
- un documento può avere molte run di analisi,
- una run può produrre un risultato,
- ogni run è collegata alle versioni dei modelli usati.

Lo schema viene inizializzato automaticamente dai file SQL presenti in `db/init`.

## Design choices

- **Batching**: riduce overhead per richiesta e migliora il throughput complessivo.
- **Retry con backoff e jitter**: aiuta a gestire errori HTTP transitori senza sovraccaricare il servizio di inferenza.
- **Servizio NLP separato**: isola dipendenze ML e runtime applicativo del backend.
- **Persistenza di `modelversion`**: migliora tracciabilità e riproducibilità delle analisi.
- **Verifica post-persistenza**: aggiunge un controllo di sicurezza prima di confermare il batch.
- **Summarization condizionale**: limita l'uso dell'LLM ai soli documenti che ne hanno davvero bisogno.

## Semplificazioni

- L'ingresso usa una mock queue invece di un broker reale.
- I dati vengono caricati da un file JSON locale.
- Autenticazione, autorizzazione e scenari multi-tenant non sono implementati.
- L'osservabilità è minima: ci sono log, ma non una stack completa di metriche o tracing.
- La logica di retry è locale all'orchestratore e volutamente semplice.
- L'API NLP è sincrona dal punto di vista del chiamante, anche se internamente il servizio Python usa async per alcune parti.
- La gestione delle versioni dei modelli è leggera e orientata alla riproducibilità.
- Il meccanismo di failure injection è pensato per test e debug, non per produzione.

## Struttura del repository

```text
.
├── compose.yml
├── backend/
│   ├── main.go
│   ├── orchestrator.go
│   ├── batch.go
│   ├── store.go
│   └── failure_injection.go
├── nlp_service/
│   ├── app.py
│   ├── nlp_inference.py
│   └── requirements.txt
└── test_data.json
```

## API

Il servizio NLP espone due endpoint:
- `POST /analyze` per l'inferenza batch
- `GET /health` per i readiness check

Il backend usa `/analyze` come endpoint di inferenza remoto.
