package main

import (
	"context"
	"fmt"
	"log"
	"os"

	qdrant "github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
)

const (
	CollectionName = "demo_docs"
	QdrantHost     = "localhost"
	QdrantPort     = 6334
)

var documents = []string{
	"The latest iPhone model comes with impressive features and a powerful camera.",
	"Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
	"Einstein's theory of relativity revolutionized our understanding of space and time.",
	"Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
	"The American Revolution had a profound impact on the birth of the United States as a nation.",
	"Regular exercise and a balanced diet are essential for maintaining good physical health.",
	"Leonardo da Vinci's Mona Lisa is considered one of the most iconic paintings in art history.",
	"Climate change poses a significant threat to the planet's ecosystems and biodiversity.",
	"Startup companies often face challenges in securing funding and scaling their operations.",
	"Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'",
}
var genres = []string{
	"technology",
	"travel",
	"science",
	"food",
	"history",
	"fitness",
	"art",
	"climate change",
	"business",
	"music",
}

type Application struct {
	qdrant *qdrant.Client
}

func main() {
	client, err := qdrant.NewClient(&qdrant.Config{
		Host: QdrantHost,
		Port: QdrantPort,
	})
	if err != nil {
		log.Fatal(err)
	}

	app := &Application{
		qdrant: client,
	}

	// Run only if data hasn't been persisted already
	// app.embedVectorsAndStoreInDB(documents, genres)

	query := "Tell me about some delicious food"
	app.queryQdrant(query)
}

func (app *Application) queryQdrant(query string) {
	ctx := context.Background()

	embedder := getEmbedder()

	queryVec, err := embedder.EmbedQuery(ctx, query)
	if err != nil {
		log.Fatalf("failed to embed query: %v", err)
	}

	results, err := app.qdrant.GetPointsClient().Search(ctx, &qdrant.SearchPoints{
		CollectionName: CollectionName,
		Vector:         queryVec,
		Limit:          2,
		WithPayload:    qdrant.NewWithPayload(true),
	})
	if err != nil {
		log.Fatalf("search failed: %v", err)
	}

	for _, result := range results.Result {
		fmt.Println("RESULT")
		for key, value := range result.Payload {
			fmt.Printf("Key: %s\nValue: %v\n", key, value)
		}
		fmt.Printf("Result (score=%.4f)\nText: %s\nGenre: %s\n", result.Score, result.Payload["text"], result.Payload["genre"])
		fmt.Println()
	}

}

func (app *Application) embedVectorsAndStoreInDB(documents, genres []string) {
	ctx := context.Background()

	embedder := getEmbedder()

	vectors, err := embedder.EmbedDocuments(ctx, documents)
	if err != nil {
		log.Fatalf("failed to embed documents: %v", err)
	}

	vectorSize := len(vectors[0])

	err = app.qdrant.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: CollectionName,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     uint64(vectorSize),
			Distance: qdrant.Distance_Cosine,
		}),
	})
	if err != nil {
		log.Println("collection may already exist:", err)
	}

	points := make([]*qdrant.PointStruct, 0, len(documents))

	for i := range documents {
		points = append(points, &qdrant.PointStruct{
			Id:      qdrant.NewIDNum(uint64(i)),
			Vectors: qdrant.NewVectors(vectors[i]...),
			Payload: qdrant.NewValueMap(map[string]any{
				"genre": genres[i],
				"text":  documents[i],
			}),
		})
	}

	_, err = app.qdrant.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: CollectionName,
		Points:         points,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Documents embedded and stored in Qdrant")
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
