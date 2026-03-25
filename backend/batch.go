package main

import (
	"sync"
	"time"
)

// DocumentStatus represents the processing status of a document.
type DocumentStatus uint8

const (
	StatusPending    DocumentStatus = iota
	StatusProcessing
	StatusDone
	StatusFailed
)

// Batch contains a set of documents to be processed together.
//
// "Struct of arrays" design: instead of []Document (array of structs, each
// element with all its fields), we use parallel fields. This reduces
// allocation because:
//   - a single contiguous backing array per field
//   - better cache locality during access to a single field (e.g., only texts)
//   - no wasted padding/alignment between fields of different types
//
// All slices are pre-allocated with the batch capacity at creation time:
// no append = no dynamic reallocation.
//
// Concurrency contract:
//   - SetDocument must be called before any Worker starts (single-writer phase).
//   - SetResult and SetError may be called concurrently by Workers, each on a
//     DISTINCT index. The caller is responsible for index partitioning.
//   - FailedIndices and IsDone must be called only after all Workers have
//     completed (or under external synchronization).
type Batch struct {
	// mu protects Statuses and ProcessingErrors during concurrent Worker writes.
	// RWMutex allows multiple concurrent readers (FailedIndices, IsDone) while
	// still serializing writes from concurrent Workers calling SetError.
	mu sync.RWMutex

	// Batch metadata.
	ID        string
	CreatedAt time.Time
	UpdatedAt time.Time // last modification time; useful for retry debugging
	Size      int       // number of documents in the batch

	// Parallel fields: IDs[i], Texts[i], SourceIDs[i] refer to the
	// same document. Index access is O(1) and cache-friendly.
	IDs       []string // document UUIDs
	SourceIDs []string // FK to source_metadata
	Texts     []string // original text (the heaviest field in memory)

	// Results: written by Workers concurrently.
	// We use fixed-size types where possible (float32 < float64).
	SentimentScores  []float32      // [-1.0, 1.0]
	SentimentLabels  []string       // "Positive", "Negative", "Neutral"
	Entities         [][]Entity     // slice of entities per document
	Summaries        []string       // only for negative documents
	Statuses         []DocumentStatus
	ProcessingErrors []string // serialised error strings; avoids []error GC pressure

	// Traceability of the models used for this batch.
	ModelVersions BatchModelVersions
}

// Entity represents a named entity extracted from the text.
// It is a small value type: do not use pointers to avoid GC pressure.
type Entity struct {
	Text  string
	Label string  // PER, ORG, LOC, MISC
	Start int32   // character offset
	End   int32
	Score float32
}

// BatchModelVersions tracks which model version was used
// for each task in this batch. Fundamental for reproducibility.
type BatchModelVersions struct {
	SentimentModelID string
	NERModelID       string
	SummaryModelID   string
}

// NewBatch creates a Batch pre-allocated for size documents.
// Always pass the exact size: avoids internal slice reallocations.
func NewBatch(id string, size int) *Batch {
	now := time.Now().UTC()
	return &Batch{
		ID:        id,
		CreatedAt: now,
		UpdatedAt: now,
		Size:      size,

		// Pre-allocation with exact capacity: cap == len == size.
		// make([]T, size) allocates and zeroes in a single syscall.
		IDs:       make([]string, size),
		SourceIDs: make([]string, size),
		Texts:     make([]string, size),

		SentimentScores:  make([]float32, size),
		SentimentLabels:  make([]string, size),
		Entities:         make([][]Entity, size),
		Summaries:        make([]string, size),
		Statuses:         make([]DocumentStatus, size),
		ProcessingErrors: make([]string, size),
	}
}

// SetDocument populates the i-th slot of the batch.
//
// Must be called before any Worker goroutine starts (single-writer phase).
// Concurrent calls from multiple goroutines are NOT safe; index partitioning
// alone does not protect the UpdatedAt field shared across all indices.
func (b *Batch) SetDocument(i int, id, sourceID, text string) {
	b.IDs[i] = id
	b.SourceIDs[i] = sourceID
	b.Texts[i] = text
	b.Statuses[i] = StatusPending
}

// SetResult writes the inference result for document i.
//
// Safe to call concurrently from multiple Workers provided each Worker owns a
// DISTINCT set of indices. Writing different indices of the same slice is safe
// in Go (no false sharing at the language level). UpdatedAt is updated under
// the write lock to avoid a data race on the shared timestamp field.
func (b *Batch) SetResult(i int, score float32, label string, entities []Entity, summary string) {
	// Write result fields first — visible to readers only after StatusDone is set.
	b.SentimentScores[i] = score
	b.SentimentLabels[i] = label
	b.Entities[i] = entities
	b.Summaries[i] = summary

	// Acquire the lock only for the shared fields that require serialization:
	// Statuses (memory-ordering guarantee for the status sentinel) and UpdatedAt.
	b.mu.Lock()
	b.Statuses[i] = StatusDone
	b.UpdatedAt = time.Now().UTC()
	b.mu.Unlock()
}

// SetError marks document i as failed. Thread-safe.
func (b *Batch) SetError(i int, err error) {
	b.mu.Lock()
	b.ProcessingErrors[i] = err.Error()
	b.Statuses[i] = StatusFailed
	b.UpdatedAt = time.Now().UTC()
	b.mu.Unlock()
}

// FailedIndices returns the indices of failed documents.
// Used by the Orchestrator for retries.
//
// Must be called after all Workers have completed, or the caller must ensure
// no concurrent writes to Statuses are in flight.
func (b *Batch) FailedIndices() []int {
	b.mu.RLock()
	defer b.mu.RUnlock()

	// Capacity heuristic: expect at most 25% failures, minimum 1 to avoid
	// zero-cap allocation on small batches (Size < 4).
	cap := b.Size / 4
	if cap == 0 {
		cap = b.Size
	}
	failed := make([]int, 0, cap)
	for i, s := range b.Statuses {
		if s == StatusFailed {
			failed = append(failed, i)
		}
	}
	return failed
}

// IsDone reports whether all documents have reached a terminal state
// (StatusDone or StatusFailed). Safe to poll concurrently.
func (b *Batch) IsDone() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()

	for _, s := range b.Statuses {
		if s == StatusPending || s == StatusProcessing {
			return false
		}
	}
	return true
}

// TextsSlice returns only the texts of the specified indices,
// without allocating a new complete Batch structure.
// Useful for reprocessing only failed documents.
func (b *Batch) TextsSlice(indices []int) []string {
	out := make([]string, len(indices))
	for j, i := range indices {
		out[j] = b.Texts[i]
	}
	return out
}
