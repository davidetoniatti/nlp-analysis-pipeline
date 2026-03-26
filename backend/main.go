package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
)

func loadMockData(filename string) ([]Message, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("apertura file fallita: %w", err)
	}
	defer file.Close()

	var data []struct {
		ID       string `json:"ID"`
		SourceID string `json:"SourceID"`
		Text     string `json:"Text"`
	}

	if err := json.NewDecoder(file).Decode(&data); err != nil {
		return nil, fmt.Errorf("decodifica JSON fallita: %w", err)
	}

	messages := make([]Message, len(data))
	for i, d := range data {
		messages[i] = Message{
			ID:       d.ID,
			SourceID: d.SourceID,
			Text:     d.Text,
		}
	}
	return messages, nil
}

func resolveMockDataPath() string {
	if envPath := os.Getenv("MOCK_DATA_FILE"); envPath != "" {
		return envPath
	}

	candidates := []string{
		"test_data.json",
		filepath.Join("..", "test_data.json"),
	}

	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}

	return "test_data.json"
}

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	cfg := DefaultConfig()
	if envURL := os.Getenv("INFERENCE_URL"); envURL != "" {
		cfg.InferenceURL = envURL
	}

	dataFile := resolveMockDataPath()

	logger.Info(
		"avvio sistema",
		"inference_url", cfg.InferenceURL,
		"workers", cfg.NumWorkers,
		"batch_size", cfg.BatchSize,
		"inference_batch_size", cfg.InferenceBatchSize,
		"data_file", dataFile,
	)

	messages, err := loadMockData(dataFile)
	if err != nil {
		logger.Error("impossibile caricare i dati di test", "err", err)
		os.Exit(1)
	}
	logger.Info("dati mock caricati", "totale_documenti", len(messages))

	queue := NewMockQueue(messages)
	aiClient := NewHTTPInferenceClient(cfg.InferenceURL)

	resultsCh := make(chan ProcessedBatch, 100)
	orch := NewOrchestrator(cfg, queue, aiClient, resultsCh, logger)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var persistWG sync.WaitGroup
	persistWG.Add(1)

	go func() {
		defer persistWG.Done()

		processedDocs := 0

		for item := range resultsCh {
			batch := item.Batch

			fmt.Printf("\n=== ANALISI BATCH: %s ===\n", batch.ID)
			fmt.Printf("Modelli: sentiment=%s | ner=%s | summary=%s\n",
				batch.ModelVersions.SentimentModelID,
				batch.ModelVersions.NERModelID,
				batch.ModelVersions.SummaryModelID,
			)

			for i := 0; i < batch.Size; i++ {
				processedDocs++

				fmt.Printf("\n[%d] ID: %s\n", processedDocs, batch.IDs[i])
				fmt.Printf("  SourceID: %s\n", batch.SourceIDs[i])
				fmt.Printf("  Testo: %s\n", batch.Texts[i])
				fmt.Printf("  Status: %d\n", batch.Statuses[i])
				fmt.Printf("  Sentiment: %s (%.2f)\n", batch.SentimentLabels[i], batch.SentimentScores[i])
				fmt.Printf("  ProcessingMs: %d\n", batch.ProcessingMs[i])

				if len(batch.Entities[i]) > 0 {
					fmt.Printf("  Entità: ")
					for _, e := range batch.Entities[i] {
						fmt.Printf("[%s: %s (%.2f)] ", e.Text, e.Label, e.Score)
					}
					fmt.Println()
				} else {
					fmt.Printf("  Entità: nessuna rilevata\n")
				}

				if batch.Summaries[i] != "" {
					fmt.Printf("  Summary:\n    %s\n", batch.Summaries[i])
				}

				if batch.ProcessingErrors[i] != "" {
					fmt.Printf("  Errore: %s\n", batch.ProcessingErrors[i])
				}
			}

			logger.Info("persisting batch", "batch_id", batch.ID, "size", batch.Size)

			// Demo mode: assume persistence succeeds.
			persistOK := true

			if persistOK {
				if err := item.Ack(); err != nil {
					logger.Error("ack batch fallito", "batch_id", batch.ID, "err", err)
				} else {
					logger.Info(
						"batch persistito e ackato",
						"batch_id", batch.ID,
						"totale_processati", processedDocs,
					)
				}
			} else {
				if err := item.Nack(); err != nil {
					logger.Error("nack batch fallito", "batch_id", batch.ID, "err", err)
				} else {
					logger.Warn("batch non persistito, inviato in retry", "batch_id", batch.ID)
				}
			}
		}
	}()

	logger.Info("orchestrator in esecuzione...")

	if err := orch.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		logger.Error("orchestrator terminato con errore", "err", err)
		close(resultsCh)
		persistWG.Wait()
		os.Exit(1)
	}

	close(resultsCh)
	persistWG.Wait()
	logger.Info("spegnimento completato con successo")
}
