"""
NLP inference service: sentiment (multilingual XLM-RoBERTa), NER (multilingual),
conditional summarization (via OpenAI-compatible API) with structured JSON output.

Dependencies:
    pip install transformers torch accelerate openai python-dotenv pydantic
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
DEFAULT_NER_MODEL = "Babelscape/wikineural-multilingual-ner"
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
PROMPT_VERSION = "v2"

DEFAULT_MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "4000"))
DEFAULT_LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "20"))
ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"


# Data structures
@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float


@dataclass
class AnalysisResult:
    doc_id: str
    sentiment_score: float = 0.0       # signed confidence in [-1.0, 1.0]
    sentiment_label: str = "Neutral"   # Positive | Negative | Neutral
    entities: list[Entity] = field(default_factory=list)
    summary: Optional[str] = None      # only if sentiment == Negative
    error: Optional[str] = None


@dataclass
class ModelVersionInfo:
    sentiment_model: str
    ner_model: str
    summary_model: str
    prompt_version: str = PROMPT_VERSION


class SummaryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=200)
    urgency: Literal["critica", "alta", "media"]
    suggested_action: str = Field(min_length=1, max_length=200)


def normalize_text(text: str, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars]


def merge_error(existing: Optional[str], new_error: Optional[str]) -> Optional[str]:
    if not new_error:
        return existing
    if not existing:
        return new_error
    return f"{existing}; {new_error}"


# Sentiment model with dynamic batching

class SentimentAnalyzer:
    """
    Wraps XLM-RoBERTa for multilingual sentiment analysis.
    Supports dynamic batching: groups texts by length (bucketing)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SENTIMENT_MODEL,
        max_batch_tokens: int = 6000,
    ):
        self.device = 0 if torch.cuda.is_available() else -1
        self.max_batch_tokens = max_batch_tokens
        device_name = "CUDA" if self.device == 0 else "CPU"
        logger.info("Loading sentiment model on %s ...", device_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Use torch.compile only if explicitly enabled: safer for cold start / portability.
        if ENABLE_TORCH_COMPILE and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                logger.info("torch.compile enabled for sentiment model.")
            except Exception:
                logger.exception("torch.compile failed, continuing without it.")

        # HF pipeline
        self._pipe: Pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=self.tokenizer,
            device=self.device,
            return_all_scores=False,
        )
        self.model_name = model_name
        logger.info("Sentiment model ready.")

    def _bucket_sort(self, texts: list[str]) -> list[tuple[int, str, int]]:
        """
        Sorts texts by approximate token count.
        Returns a list of (original_index, text, token_len).
        """
        if not texts:
            return []

        tokenized = self.tokenizer(
            texts,
            add_special_tokens=True,
        )
        indexed = [
            (i, text, len(input_ids))
            for i, (text, input_ids) in enumerate(zip(texts, tokenized["input_ids"]))
        ]
        indexed.sort(key=lambda x: x[2])
        return indexed

    def _build_dynamic_batches(
        self,
        sorted_items: list[tuple[int, str, int]],
        max_batch_size: int,
    ) -> list[list[tuple[int, str, int]]]:
        batches: list[list[tuple[int, str, int]]] = []
        current: list[tuple[int, str, int]] = []
        current_tokens = 0

        for item in sorted_items:
            _, _, token_len = item
            if current and (
                len(current) >= max_batch_size
                or current_tokens + token_len > self.max_batch_tokens
            ):
                batches.append(current)
                current = []
                current_tokens = 0

            current.append(item)
            current_tokens += token_len

        if current:
            batches.append(current)

        return batches

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> tuple[list[Optional[dict]], list[Optional[str]]]:
        """
        Batch inference with length-aware dynamic batching.
        Returns:
            predictions in original order and optional per-document errors.
        """
        if not texts:
            return [], []

        sorted_items = self._bucket_sort(texts)
        dynamic_batches = self._build_dynamic_batches(sorted_items, max_batch_size=batch_size)

        predictions: list[Optional[dict]] = [None] * len(texts)
        errors: list[Optional[str]] = [None] * len(texts)

        t0 = time.perf_counter()

        for batch in dynamic_batches:
            chunk_indices = [i for i, _, _ in batch]
            chunk_texts = [t for _, t, _ in batch]

            try:
                with torch.inference_mode():
                    chunk_results = self._pipe(
                        chunk_texts,
                        batch_size=len(chunk_texts),
                    )
                for idx, pred in zip(chunk_indices, chunk_results):
                    predictions[idx] = pred
            except Exception as batch_exc:
                logger.exception("Sentiment chunk failed, retrying per document: %s", batch_exc)
                for idx, text, _ in batch:
                    try:
                        with torch.inference_mode():
                            pred = self._pipe(
                                text,
                                batch_size=1,
                            )
                        predictions[idx] = pred
                    except Exception as doc_exc:
                        errors[idx] = str(doc_exc)

        elapsed = time.perf_counter() - t0
        throughput = len(texts) / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Sentiment inference: %d texts in %.2fs (%.1f text/s)",
            len(texts),
            elapsed,
            throughput,
        )
        return predictions, errors

    def score_to_label(self, raw_label: str, score: float) -> tuple[str, float]:
        """
        Converts model output to normalized label and score in [-1, 1].
        (HF score is the confidence of the predicted class)
        """
        l = (raw_label or "").lower()
        if "neg" in l:
            return "Negative", -float(score)
        if "pos" in l:
            return "Positive", float(score)
        return "Neutral", 0.0


# Named Entity Recognition
class NERExtractor:
    """
    Extracts named entities with a multilingual NER model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NER_MODEL,
        score_threshold: float = 0.80,
    ):
        self.device = 0 if torch.cuda.is_available() else -1
        self.score_threshold = score_threshold

        logger.info("Loading NER model on %s ...", "CUDA" if self.device == 0 else "CPU")
        self._pipe: Pipeline = pipeline(
            "ner",
            model=model_name,
            device=self.device,
            aggregation_strategy="simple",
        )
        self.model_name = model_name
        logger.info("NER model ready.")

    def extract_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> tuple[list[list[Entity]], list[Optional[str]]]:
        results: list[list[Entity]] = [[] for _ in texts]
        errors: list[Optional[str]] = [None] * len(texts)

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            chunk_indices = list(range(start, min(start + batch_size, len(texts))))

            try:
                with torch.inference_mode():
                    raw = self._pipe(
                        chunk,
                        batch_size=len(chunk),
                    )

                for idx, doc_entities in zip(chunk_indices, raw):
                    results[idx] = self._convert_entities(doc_entities)
            except Exception as batch_exc:
                logger.exception("NER chunk failed, retrying per document: %s", batch_exc)
                for idx, text in zip(chunk_indices, chunk):
                    try:
                        with torch.inference_mode():
                            doc_raw = self._pipe(
                                text,
                                batch_size=1,
                            )
                        results[idx] = self._convert_entities(doc_raw)
                    except Exception as doc_exc:
                        errors[idx] = str(doc_exc)
                        results[idx] = []

        return results, errors

    def _convert_entities(self, raw_entities: list[dict]) -> list[Entity]:
        entities: list[Entity] = []
        for e in raw_entities:
            score = float(e.get("score", 0.0))
            if score < self.score_threshold:
                continue
            entities.append(
                Entity(
                    text=str(e.get("word", "")),
                    label=str(e.get("entity_group", "")),
                    start=int(e.get("start", 0)),
                    end=int(e.get("end", 0)),
                    score=score,
                )
            )
        return entities


# Summarizer with conditional logic (only for negative texts)
class ConditionalSummarizer:
    """
    Generates a summary only if sentiment is Negative.
    """

    MAX_CONCURRENT_REQUESTS = 10

    _SYSTEM_PROMPT = (
        "Sei un assistente specializzato nell'analisi di feedback negativi dei clienti. "
        "Analizza internamente il testo e restituisci SOLO il JSON finale richiesto, "
        "senza spiegazioni aggiuntive."
    )

    _USER_PROMPT_TEMPLATE = """Hai ricevuto il seguente feedback negativo di un cliente:

<testo>
{text}
</testo>

Entità rilevate nel testo: {entities}

Obiettivo:
- sintetizza il problema principale;
- stima l'urgenza: critica, alta oppure media;
- suggerisci un'azione immediata per il team di supporto.

Restituisci ESCLUSIVAMENTE un oggetto JSON con questa struttura:
{{
  "summary": "<riassunto del problema in max 20 parole>",
  "urgency": "<critica|alta|media>",
  "suggested_action": "<azione specifica in max 20 parole>"
}}
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_LLM_MODEL,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
        timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
    ):
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError("GROQ_API_KEY not set")

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=GROQ_BASE_URL,
        )
        self._model = model
        self._timeout_s = timeout_s
        self._max_text_chars = max_text_chars

        # A maximum of max_concurrent coroutines access the API simultaneously. The others wait without blocking.
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def summarize(
        self,
        text: str,
        entities: list[Entity],
        sentiment_label: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Generates the summary only if sentiment is Negative.
        Returns:
            (normalized_json_string, error_message)
        """
        if sentiment_label != "Negative":
            return None, None  # no API call for non-negative texts

        safe_text = normalize_text(text, max_chars=self._max_text_chars)
        entity_str = ", ".join(f"{e.text} ({e.label})" for e in entities[:25]) or "nessuna"
        prompt = self._USER_PROMPT_TEMPLATE.format(text=safe_text, entities=entity_str)

        async with self._semaphore:  # maximum MAX_CONCURRENT_REQUESTS at a time
            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": self._SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                        max_tokens=200,
                        response_format={"type": "json_object"},
                    ),
                    timeout=self._timeout_s,
                )

                raw = (response.choices[0].message.content or "").strip()
                if not raw:
                    return None, "empty_summary_response"

                try:
                    parsed = json.loads(raw)
                    payload = SummaryPayload.model_validate(parsed)
                    normalized_json = json.dumps(payload.model_dump(), ensure_ascii=False)
                    return normalized_json, None
                except json.JSONDecodeError as exc:
                    logger.error("Invalid JSON from summarization: %s | raw: %.200s", exc, raw)
                    return None, "invalid_summary_json"
                except ValidationError as exc:
                    logger.error("Summary schema validation failed: %s | raw: %.200s", exc, raw)
                    return None, "invalid_summary_schema"

            except asyncio.TimeoutError:
                logger.error("Summarization timeout after %.1fs", self._timeout_s)
                return None, "summary_timeout"
            except Exception as exc:
                logger.error("Summarization error: %s", exc)
                return None, str(exc)

    async def summarize_batch(
        self,
        texts: list[str],
        entities_list: list[list[Entity]],
        labels: list[str],
    ) -> tuple[list[Optional[str]], list[Optional[str]]]:
        """
        Processes only negative texts in parallel.
        """
        summaries: list[Optional[str]] = [None] * len(texts)
        errors: list[Optional[str]] = [None] * len(texts)

        tasks: dict[int, asyncio.Task[tuple[Optional[str], Optional[str]]]] = {}
        for i, (text, ents, label) in enumerate(zip(texts, entities_list, labels)):
            if label == "Negative":
                tasks[i] = asyncio.create_task(self.summarize(text, ents, label))

        if not tasks:
            return summaries, errors

        completed = await asyncio.gather(*tasks.values())
        for idx, (summary, error) in zip(tasks.keys(), completed):
            summaries[idx] = summary
            errors[idx] = error

        return summaries, errors


# Final pipeline
class NLPPipeline:
    """
    Coordinates sentiment, NER, and summarization in a single pipeline.
    """

    def __init__(
        self,
        sentiment_model: str = DEFAULT_SENTIMENT_MODEL,
        ner_model: str = DEFAULT_NER_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
    ):
        self.sentiment = SentimentAnalyzer(sentiment_model)
        self.ner = NERExtractor(ner_model)
        self.summarizer = ConditionalSummarizer(model=llm_model)

    async def process_batch(
        self,
        doc_ids: list[str],
        texts: list[str],
        batch_size: int = 32,
    ) -> tuple[list[AnalysisResult], ModelVersionInfo]:
        """
        Processes a full batch: sentiment → NER → conditional summarization.
        Returns results in the same order as input.
        """
        if len(doc_ids) != len(texts):
            raise ValueError("Mismatched doc_ids and texts")
        if not doc_ids:
            raise ValueError("Empty batch")

        prepared_texts = [normalize_text(t) for t in texts]
        empty_positions = [i for i, t in enumerate(prepared_texts) if not t]
        if empty_positions:
            raise ValueError(f"Empty texts at positions: {empty_positions}")

        n = len(prepared_texts)
        results = [AnalysisResult(doc_id=doc_id) for doc_id in doc_ids]
        logger.info("Processing %d documents.", n)

        # Sentiment (blocking HF pipeline offloaded to a worker thread)
        raw_sentiments, sentiment_errors = await asyncio.to_thread(
            self.sentiment.predict_batch,
            prepared_texts,
            batch_size,
        )

        labels = ["Neutral"] * n
        for i, raw in enumerate(raw_sentiments):
            if raw is None:
                results[i].error = merge_error(
                    results[i].error,
                    f"sentiment_failed: {sentiment_errors[i] or 'unknown_error'}",
                )
                continue

            label, score = self.sentiment.score_to_label(raw.get("label", ""), float(raw.get("score", 0.0)))
            labels[i] = label
            results[i].sentiment_label = label
            results[i].sentiment_score = score

        # NER (blocking HF pipeline offloaded to a worker thread)
        entities_list, ner_errors = await asyncio.to_thread(
            self.ner.extract_batch,
            prepared_texts,
            batch_size,
        )

        for i, entities in enumerate(entities_list):
            results[i].entities = entities
            if ner_errors[i]:
                results[i].error = merge_error(results[i].error, f"ner_failed: {ner_errors[i]}")

        # Conditional summarization (async, only negative texts)
        summaries, summary_errors = await self.summarizer.summarize_batch(
            prepared_texts,
            entities_list,
            labels,
        )

        for i, summary in enumerate(summaries):
            results[i].summary = summary
            if summary_errors[i]:
                results[i].error = merge_error(results[i].error, f"summary_failed: {summary_errors[i]}")

        neg_count = sum(1 for label in labels if label == "Negative")
        logger.info(
            "Batch completed: %d positives, %d negatives, %d neutral",
            labels.count("Positive"),
            neg_count,
            labels.count("Neutral"),
        )

        return results, ModelVersionInfo(
            sentiment_model=self.sentiment.model_name,
            ner_model=self.ner.model_name,
            summary_model=self.summarizer._model,
            prompt_version=PROMPT_VERSION,
        )


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------

async def main():
    pipeline_instance = NLPPipeline()

    sample_ids = ["doc-001", "doc-002", "doc-003"]
    sample_texts = [
        "Ottimo prodotto, sono molto soddisfatto dell'acquisto!",
        "Il servizio clienti di Acme Corp è stato terribile. Ho aspettato 3 settimane e il pacco non è mai arrivato.",
        "Prodotto nella media, niente di speciale.",
    ]

    results, model_info = await pipeline_instance.process_batch(sample_ids, sample_texts)

    print(model_info)
    for r in results:
        print(f"\n--- {r.doc_id} ---")
        print(f"  Sentiment: {r.sentiment_label} ({r.sentiment_score:+.3f})")
        print(f"  Entità: {[e.text for e in r.entities]}")
        print(f"  Error: {r.error}")
        if r.summary:
            print(f"  Summary JSON:\n{r.summary}")


if __name__ == "__main__":
    asyncio.run(main())
