# Scalable NLP Analysis Pipeline

For each input document, the system:
- collects the text (multilingual: Italian and English),
- performs sentiment analysis,
- extracts named entities (NER),
- generates a summary (via LLM) only for texts with negative sentiment,
- persistently saves runs and results in the database to ensure traceability and reproducibility.

## Architecture

- **Postgres**: saves documents, analysis runs, results, source metadata, and model versions (minimal model registry).
- **Go Backend (Orchestrator)**: consumes messages from a mock queue, builds batches, calls the NLP service, manages retries with exponential backoff, and persists results while verifying their integrity.
- **NLP Service (Python/FastAPI)**: exposes an HTTP API that performs sentiment analysis and NER locally (Transformers) and conditional summarization via Groq/LLM.

## Deployment and Startup

### Requirements

- Docker
- Docker Compose
- A valid `GROQ_API_KEY` (optional, for summarization)

### Startup

```bash
docker compose up --build
```

### Configuration

- `DATABASE_URL`: Postgres connection URL for the Go backend.
- `INFERENCE_URL`: NLP service endpoint (default: `http://ai_service:8080/analyze`).
- `GROQ_API_KEY`: API key for summarization via LLM.
- `FAIL_DOC_IDS`: List of document IDs (comma-separated) to simulate injected failures.
- `FAIL_TEXT_CONTAINS`: String that, if contained in the text, triggers a simulated failure.
- `REQUEST_TIMEOUT_S`: Timeout for requests to the NLP service.
- `LLM_TIMEOUT_S`: Specific timeout for the LLM call.

## Data Flow

1. The Go backend loads mock documents from `test_data.json`.
2. Documents are grouped into batches based on size or time (flush interval).
3. The backend calls the `/analyze` endpoint of the Python service.
4. The NLP service performs:
   - Multilingual sentiment analysis (XLMR-Roberta).
   - Multilingual NER (WikiNeural).
   - Conditional summarization (Llama 3 via Groq) only for negative texts.
5. The backend receives the batch results and metadata for the model versions used.
6. Results are persisted in Postgres within a single transaction.
7. Persistence is verified by querying the database before acknowledging (ack) the batch.

## Database Schema

The database is designed to support full traceability of every analysis.

Tables:

- `source_metadata`: Information about the document's origin.
- `document`: The raw input document received.
- `model_version`: Registry of models used (name, version, provider, prompt hash).
- `analysis_run`: Record of an analysis execution, linked to the document and models used.
- `analysis_result`: The structured output (sentiment, entities, summary) of a completed run.


## Design Choices

- **Batching**: Optimizes throughput and reduces the number of network calls.
- **Retry with backoff and jitter**: Handles transient errors and backpressure (429) resiliently.
- **Model Traceability**: Each result is linked to a specific `model_version`, including model revision and prompt hash.

## Repository Structure

```text
.
├── compose.yml
├── test_data.json
├── backend/
│   ├── cmd/server/       # Application entry point
│   ├── internal/pipeline/# batching, store, orchestrator
│   ├── Dockerfile
│   └── go.mod
├── nlp_service/          # NLP pipeline
│   ├── app.py
│   ├── nlp_inference.py
│   └── requirements.txt
└── db/init/
```

## API

The NLP service exposes:
- `POST /analyze`: Accepts batches of documents and returns analysis and model metadata.
- `GET /health`: Verifies service readiness and model loading.

