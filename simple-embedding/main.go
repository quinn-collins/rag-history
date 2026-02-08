package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
)

// Word2Vec was a successful vectorization algorithm, you can download other peoples vectors that have used this vectorization such as google news.
// There are other pre-trained embedding models: E.g.
//	   ['fasttext-wiki-news-subwords-300',
//	   'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300',
//	   'word2vec-google-news-300', 'glove-wiki-gigaword-50',
//	   'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200',
//	   'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50',
//	   'glove-twitter-100', 'glove-twitter-200',
//	   '__testing_word2vec-matrix-synopsis']

type Match struct {
	Index int
	Score float64
}

func main() {
	ctx := context.Background()
	fmt.Println(ctx)

	// Read file into a byte slice
	content, err := os.ReadFile("./corpora/shakespeare-macbeth.txt")
	if err != nil {
		log.Fatalf("failed to read in documents: %v", err)
	}

	// Set up documents for embedding
	documents := strings.Split(string(content), "\n")
	var normDocuments []string
	for _, d := range documents {
		if d != "" {
			normDocuments = append(normDocuments, d)
		}
	}

	// documents := []string{
	// 	"The cat is on the mat.",
	// 	"There is a cat on the mat.",
	// 	"The dog is in the yard.",
	// 	"There is a dog in the yard.",
	// }

	// Set up query for embedding
	// query := "When should we get together again?"
	// Top Matches:
	// 	1) score=0.9036 |   1. When shall we three meet againe?
	// 	2) score=0.8353 | Our point of second meeting.
	// 	3) score=0.8308 | When shalt thou see thy wholsome dayes againe?
	// 	4) score=0.8297 | Was it not yesterday we spoke together?
	// 	5) score=0.8135 | Shall we well meet them, that way are they comming
	// query := "Will the ocean clean this blood?"
	// Top Matches:
	// 	1) score=0.9032 | Will all great Neptunes Ocean wash this blood
	// 	2) score=0.8319 | What will these hands ne're be cleane? No more o'that
	// 	3) score=0.8283 | A little Water cleares vs of this deed.
	// 	4) score=0.8155 | Blood will haue Blood:
	// 	5) score=0.8096 | Cleanse the stufft bosome, of that perillous stuffe
	query := "I thought I heard someone yell, 'No more sleep'"
	// Top Matches:
	// 	1) score=0.8607 |    Macb. Me thought I heard a voyce cry, Sleep no more:
	// 	2) score=0.8349 | Shall sleepe no more: Macbeth shall sleepe no more
	// 	3) score=0.8306 | And yet I would not sleepe:
	// 	4) score=0.8285 | Sleepe shall neyther Night nor Day
	// 	5) score=0.8259 | Hath rung Nights yawning Peale,
	// query := "A cat is sitting on a mat."

	embedder := getEmbedder()
	similarities := embedDocsAndQuery(ctx, embedder, query, normDocuments)

	matches := make([]Match, 0, len(similarities))
	for i, score := range similarities {
		matches = append(matches, Match{
			Index: i,
			Score: score,
		})
	}

	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Score > matches[j].Score
	})

	topK := 5
	if len(matches) < topK {
		topK = len(matches)
	}

	fmt.Println("\nTop Matches:")
	for i := 0; i < topK; i++ {
		m := matches[i]
		fmt.Printf("%d) score=%.4f | %s\n", i+1, m.Score, normDocuments[m.Index])
	}
}

func embedDocsAndQuery(ctx context.Context, embedder *embeddings.EmbedderImpl, query string, documents []string) []float64 {
	// Embed those documents
	documentEmbeddings, err := embedder.EmbedDocuments(ctx, documents)
	if err != nil {
		log.Fatalf("failed to embed documents: %v", err)
	}

	fmt.Println("Document embeddings:")
	for i, vec := range documentEmbeddings {
		fmt.Printf("Doc %d: len=%d, first 5 dims=%v\n", i, len(vec), vec[:5])
	}

	// Embed a query
	queryEmbedding, err := embedder.EmbedQuery(ctx, query)
	if err != nil {
		log.Fatalf("failed to embed query: %v", err)
	}

	fmt.Printf("\nQuery embedding: len=%d, first 5 dims=%v\n", len(queryEmbedding), queryEmbedding[:5])

	// Calculate and return
	similarities := querySimilarities(queryEmbedding, documentEmbeddings)

	return similarities
}

func querySimilarities(query []float32, documentEmbeddings [][]float32) []float64 {
	var results []float64
	for _, doc := range documentEmbeddings {
		results = append(results, cosineSimilarity(widenFloats(doc), widenFloats(query)))
	}

	return results
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		log.Fatal("vectors must be same length")
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func getEmbedder() *embeddings.EmbedderImpl {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatalf("OPEN_API_KEY environment variable not set")
	}

	// openai.New automatically checks OPENAI_API_KEY env var
	llm, err := openai.New(
		openai.WithModel("text-embedding-3-large"),
	)
	if err != nil {
		log.Fatalf("failed to create OpenAI client: %v", err)
	}

	embedder, err := embeddings.NewEmbedder(llm)
	if err != nil {
		log.Fatalf("failed to create an OpenAI embedding model: %v", err)
	}

	return embedder
}

func widenFloats(fs []float32) []float64 {
	result := make([]float64, 0, len(fs))
	for _, f := range fs {
		result = append(result, float64(f))
	}

	return result
}
