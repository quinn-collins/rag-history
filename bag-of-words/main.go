package main

import (
	"fmt"
	"math"
	"strings"
)

// One-hot encoding is a way of representing categorial values in a numerical way.
// We create an array that has as many elements as the number of categories.
// To represent the category we have an array that consists of 0 everywhere except the element that corresponds to the category.
// Unit of representation: a single token
// Vector: length = vocabulary size
// Values: exactly one 1, rest are 0s
// Order-aware: yes (because we keep sequence)
// Note: One-hot is used at a token level where sequence matters.

// Bag-of-words
// Unit of representation: The entire document
// Vector: length = vocabulary size
// Values: Counts of tokens or binary indicators
// Order-aware: no
// Note: Bag-of-words is used as a document level for things like topic classification and spam detection.
// Advantages
//   - Easy
//   - Interpretable
//   - Domain Adaptation
// Disadvantages
//   - Dimensionality
//   - Sparse Representation
//   - Favors long documents and common words
//   - Doesn't handle synonyms
//   - Doesn't handle polysemes
//   - Unhandled unseen words
//   - Very little capture of semantic relationships
//   - Lost word order
//   - Slow

func main() {
	doc1 := "Tokenization is the process of breaking text into words."
	doc2 := "Vocabulary is the collection of unique words."
	doc3 := "The process of tokenizing is essential in NLP."
	corpus := []string{doc1, doc2, doc3}

	// First we get our list of tokens, in this case that will be space-delimited words in our corpus
	tokens := tokenize(corpus)

	// Vector size is determined by the set of vocab in the corpus
	vocab := buildVocab(tokens)
	fmt.Println(vocab)

	// One-hot encodings for both documents
	// This gives us basis vectors at a token level.
	fmt.Println("Tokens: ", tokens[0])
	doc1OneHots := documentToOneHotSequence(tokens[0], vocab)
	printVectors(doc1OneHots)
	fmt.Println()
	fmt.Println("Tokens: ", tokens[1])
	doc2OneHots := documentToOneHotSequence(tokens[1], vocab)
	printVectors(doc2OneHots)
	fmt.Println()
	fmt.Println("Tokens: ", tokens[2])
	doc3OneHots := documentToOneHotSequence(tokens[2], vocab)
	printVectors(doc3OneHots)

	// Bag of words can be computed by summing the vectors.
	// This lets us generate a vector that represents word frequency in a document.
	bow1 := bagOfWordsFromOneHots(doc1OneHots, len(vocab))
	fmt.Println(bow1)
	bow2 := bagOfWordsFromOneHots(doc2OneHots, len(vocab))
	fmt.Println(bow2)
	bow3 := bagOfWordsFromOneHots(doc3OneHots, len(vocab))

	// Once we have bags of words generated per-document we can calculate cosine-similarity between documents.
	bow1Bow2Similarity := cosineSimilarity(sliceIntToFloat(bow1), sliceIntToFloat(bow2))
	bow1Bow3Similarity := cosineSimilarity(sliceIntToFloat(bow1), sliceIntToFloat(bow3))
	bow2Bow3Similarity := cosineSimilarity(sliceIntToFloat(bow2), sliceIntToFloat(bow3))
	fmt.Println(bow1Bow2Similarity)
	fmt.Println(bow1Bow3Similarity)
	fmt.Println(bow2Bow3Similarity)
}

// tokenize returns tokens as space-delimited slices of strings of a corpus.
func tokenize(corpus []string) [][]string {
	var tokens [][]string
	for _, doc := range corpus {
		tokens = append(tokens, strings.Split(doc, " "))
	}

	return tokens
}

// buildVocab Builds out a map of strings to an int value representing where that word is in the corpus.
func buildVocab(tokens [][]string) map[string]int {
	vocab := make(map[string]int)
	index := 0

	for _, doc := range tokens {
		for _, word := range doc {
			if _, exists := vocab[word]; !exists {
				vocab[word] = index
				index++
			}
		}
	}

	return vocab
}

// oneHot returns a vector for a word that maps the word in vector space back to the index in the vocabulary.
func oneHot(word string, vocab map[string]int) []int {
	vec := make([]int, len(vocab))
	if i, ok := vocab[word]; ok {
		vec[i] = 1
	}

	return vec
}

// documentToOneHotSequence returns a one-hot vector for every token in a document.
func documentToOneHotSequence(doc []string, vocab map[string]int) [][]int {
	var result [][]int
	for _, word := range doc {
		result = append(result, oneHot(word, vocab))
	}

	return result
}

func bagOfWordsFromOneHots(oneHots [][]int, vocabSize int) []int {
	bow := make([]int, vocabSize)

	for _, vec := range oneHots {
		for i, val := range vec {
			bow[i] += val
		}
	}

	return bow
}

func printVectors(vectors [][]int) {
	for _, vector := range vectors {
		fmt.Println(vector)
	}
}

func cosineSimilarity(a, b []float64) float64 {
	assertVectorsLengthEqual(a, b)

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func assertVectorsLengthEqual(a, b []float64) {
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}
}

func sliceIntToFloat(s []int) []float64 {
	floatSlice := make([]float64, len(s))
	for i, v := range s {
		floatSlice[i] = float64(v)
	}

	return floatSlice
}
