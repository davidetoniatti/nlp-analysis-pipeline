"""
NLP inference service: sentiment (multilingual XLM-RoBERTa), NER (multilingual),
conditional summarization (LLM via API) with Chain of Thought.

Models chosen (assuming en+ita language support):
  - Sentiment: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
  - NER: Babelscape/wikineural-multilingual-ner
  - LLM: gpt-4o-mini

Dependencies:
    pip install transformers torch accelerate openai python-dotenv
"""

from __future__ import annotations

import os
import json
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Optional
import time

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Pipeline,
)
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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
    sentiment_score: float          # [-1.0, 1.0]
    sentiment_label: str            # Positive | Negative | Neutral
    entities: list[Entity] = field(default_factory=list)
    summary: Optional[str] = None   # only if sentiment == Negative
    error: Optional[str] = None

@dataclass
class ModelVersionInfo:
    sentiment_model: str
    ner_model: str
    summary_model: str
    prompt_version: str = "v1"


# Sentiment model with dynamic batching

class SentimentAnalyzer:
    """
    Wraps XLM-RoBERTa for multilingual sentiment analysis.
    Supports dynamic batching: groups texts by length (bucketing)
    """

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    ):
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "CUDA" if self.device == 0 else "CPU"
        logger.info("Loading sentiment model on %s ...", device_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Use torch.compile if available
        if hasattr(torch, "compile"):
            model = torch.compile(model)

        # HF pipeline
        self._pipe: Pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=self.tokenizer,
            device=self.device,
            return_all_scores=False,
        )
        self.model_name = model_name
        logger.info("Sentiment model pronto.")

    def _bucket_sort(self, texts: list[str]) -> list[tuple[int, str]]:
        """
        Sorts texts by length (approximate token count).
        Returns a list of (original_index, text).
        """
        indexed = list(enumerate(texts))
        indexed.sort(key=lambda x: len(x[1]))
        return indexed

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """
        Batch inference with dynamic bucketing.
        Input:
            texts: list of texts to classify
            batch_size: size of the batch sent to the GPU in a
                        single forward pass

        Output:
            list of dicts {"label": str, "score": float} in original order
        """
        if not texts:
            return []

        # sort by length
        sorted_indexed = self._bucket_sort(texts)
        sorted_texts = [t for _, t in sorted_indexed]
        original_indices = [i for i, _ in sorted_indexed]

        # batch inference
        raw_results: list[dict] = []
        t0 = time.perf_counter()

        for start in range(0, len(sorted_texts), batch_size):
            chunk = sorted_texts[start : start + batch_size]
            with torch.inference_mode():  # disable autograd
                chunk_results = self._pipe(chunk, batch_size=len(chunk))
            raw_results.extend(chunk_results)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Sentiment inference: %d text in %.2fs (%.1f text/s)",
            len(texts), elapsed, len(texts) / elapsed,
        )

        # re-order based on original indices
        ordered = [None] * len(texts)
        for rank, orig_idx in enumerate(original_indices):
            ordered[orig_idx] = raw_results[rank]

        return ordered

    def score_to_label(self, raw_label: str, score: float) -> tuple[str, float]:
        """
        Converts model output to normalized label and score in [-1, 1].
        (HF score is the probability of the predicted label: [0.5, 1.0])
        """

        l = raw_label.lower()
        if "neg" in l:
            return "Negative", -score
        elif "pos" in l:
            return "Positive", score
        else:
            return "Neutral", 0.0


# Named Entity Recognition
class NERExtractor:
    """
    Extracts named entities with a multilingual NER model.
    """

    def __init__(self, model_name: str = "Babelscape/wikineural-multilingual-ner"):
        device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading NER model.")
        self._pipe: Pipeline = pipeline(
            "ner",
            model=model_name,
            device=device,
            aggregation_strategy="simple",
        )
        self.model_name = model_name
        logger.info("NER model ready.")

    def extract_batch(self, texts: list[str], batch_size: int = 32) -> list[list[Entity]]:
        results = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            with torch.inference_mode():
                raw = self._pipe(chunk, batch_size=len(chunk))
            for doc_entities in raw:
                entities = [
                    Entity(
                        text=e["word"],
                        label=e["entity_group"],
                        start=e["start"],
                        end=e["end"],
                        score=float(e["score"]),
                    )
                    for e in doc_entities
                    if e["score"] > 0.80  # filters low confidence entities
                ]
                results.append(entities)
        return results


# Summarizer with Chain of Thought (only for negative texts)
class ConditionalSummarizer:
    """
    Generates a summary only if sentiment is Negative.
    """

    MAX_CONCURRENT_REQUESTS = 10  # concurrent slots towards OpenAI

    _SYSTEM_PROMPT = (
        "Sei un assistente specializzato nell'analisi di feedback negativi dei clienti. "
        "Il tuo compito è produrre un riassunto strutturato in formato JSON. "
        "Ragiona passo dopo passo prima di scrivere il JSON finale."
    )

    _USER_PROMPT_TEMPLATE = """Hai ricevuto il seguente feedback negativo di un cliente:

<testo>
{text}
</testo>

Entità rilevate nel testo: {entities}

Segui questi passi di ragionamento:
1. IDENTIFICA il problema principale espresso dal cliente (1 frase).
2. VALUTA l'urgenza: critica (rischio churn), alta (insoddisfazione forte), media.
3. SUGGERISCI una risposta o azione immediata per il team di supporto.

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
        model: str = "gemini-2.5-flash",
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    ):
        self._client = AsyncOpenAI(
                api_key=api_key or os.environ.get("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/" # hardcoded (it's a prototype, right?)
                )
        self._model = model
        # A maximum of max_concurrent coroutines access the API simultaneously. The others wait without blocking.
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def summarize(
        self,
        text: str,
        entities: list[Entity],
        sentiment_label: str,
    ) -> Optional[str]:
        """
        Generates the summary only if sentiment is Negative.
        """
        if sentiment_label != "Negative":
            return None  # no API call for non-negative texts

        entity_str = ", ".join(f"{e.text} ({e.label})" for e in entities) or "nessuna"
        prompt = self._USER_PROMPT_TEMPLATE.format(text=text, entities=entity_str)

        async with self._semaphore:  # maximum MAX_CONCURRENT_REQUESTS at a time
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": self._SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content

                # try parsing here to intercept errors asap, before the payload reaches the Go backend.
                try:
                    parsed = json.loads(raw)
                    # Check for presence of expected fields
                    required = {"summary", "urgency", "suggested_action"}
                    missing = required - parsed.keys()
                    if missing:
                        logger.warning("missing fields: %s", missing)
                    return raw  # return raw JSON string to backend
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON from summarization: %s | raw: %.200s", e, raw)
                    return None

            except Exception as e:
                logger.error("Summarization error: %s", e)
                return None

    async def summarize_batch(
        self,
        texts: list[str],
        entities_list: list[list[Entity]],
        labels: list[str],
    ) -> list[Optional[str]]:
        """
        Processes all negative texts in parallel with asyncio.gather.
        """
        tasks = [
            self.summarize(text, ents, label)
            for text, ents, label in zip(texts, entities_list, labels)
        ]
        return await asyncio.gather(*tasks)


# Final pipeline
class NLPPipeline:
    """
    Coordinates sentiment, NER, and summarization in a single pipeline.
    """

    def __init__(
        self,
        sentiment_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
        ner_model: str = "Babelscape/wikineural-multilingual-ner",
        llm_model: str = "gemini-2.5-flash",
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
        results: list[AnalysisResult] = []
        n = len(texts)
        logger.info("Processing %d documents.", n)

        # Sentiment (GPU, batch)
        raw_sentiments = self.sentiment.predict_batch(texts, batch_size=batch_size)
        labels, scores = [], []
        for raw in raw_sentiments:
            label, score = self.sentiment.score_to_label(raw["label"], raw["score"])
            labels.append(label)
            scores.append(score)

        # NER (GPU, batch)
        entities_list = self.ner.extract_batch(texts, batch_size=batch_size)

        # Conditional summarization (async, only negative texts)
        summaries = await self.summarizer.summarize_batch(texts, entities_list, labels)

        # aggregate results
        for i in range(n):
            results.append(
                AnalysisResult(
                    doc_id=doc_ids[i],
                    sentiment_score=scores[i],
                    sentiment_label=labels[i],
                    entities=entities_list[i],
                    summary=summaries[i],
                )
            )

        neg_count = sum(1 for l in labels if l == "Negative")
        logger.info(
            "Batch completed: %d positives, %d negatives, %d neutral",
            labels.count("Positive"), neg_count, labels.count("Neutral"),
        )
        return results, ModelVersionInfo(
            sentiment_model=self.sentiment.model_name,
            ner_model=self.ner.model_name,
            summary_model=self.summarizer._model,
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

    for r in results:
        print(f"\n--- {r.doc_id} ---")
        print(f"  Sentiment: {r.sentiment_label} ({r.sentiment_score:+.3f})")
        print(f"  Entità: {[e.text for e in r.entities]}")
        if r.summary:
            print(f"  Summary CoT:\n{r.summary}")


if __name__ == "__main__":
    asyncio.run(main())
