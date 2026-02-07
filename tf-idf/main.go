package main

import (
	"fmt"
	"math"
	"strings"
)

// Term Frequency (TF)
// The count of a word in a document divided by the word count of the document.
// I.e. "How often does the word appear in the document?"

// Inverse Document Frequency (IDF)'
// Take the log of the number of documents in the corpus divided by the number of documents containing the term.
// i.e. "How rare is this word across documents?"

// TF-IDF
// The product of the term frequency and the inverse document frequency.
// Words in only one document get boosted by their IDF score and standout in the TF_IDF score for that document.
// A word belonging to a few documents in the corpus would have a high TF_IDF for those documents.
// A word belonging to nearly all of the documents in the corpus would have a very low TF_IDF for those documents.
// Advantages
//   - Easy
//   - Interpretable
//   - Domain Adaptation
//   - Reduce the weight of high-frequency words
// Disadvantages
//   - Dimensionality
//   - Doesn't handle synonyms
//   - Doesn't handle polysemes
//   - Unhandled unseen words
//   - Very little capture of semantic relationships
//   - Lost word order
//   - Slow

func main() {
	doc1 := "My dog is the best dog that ever was a pet dog"
	doc2 := "Vocabulary is the collection of unique words."
	doc3 := "The process of tokenizing is essential in NLP."
	corpus := []string{doc1, doc2, doc3}

	// As always, tokenize and build out a dictionary of vocabulary
	tokens := tokenize(corpus)
	vocab := buildVocab(tokens)

	fmt.Println("Vocabulary:", vocab)

	// Build a slice of ints that represents how many times each word shows up in our vocab
	df := documentFrequency(tokens, vocab)
	fmt.Println("\nDF:", df)

	// Build a slice of floats that represents how rare all of our words are across all of our documents
	idf := inverseDocumentFrequency(sliceIntToFloat(df), len(corpus))
	fmt.Println("\nIDF:", idf)

	// Calculate tf_idf for each document
	for i, doc := range tokens {
		// get a frequency list for each word according to our corpus vocab
		tf := termFrequency(doc, vocab)
		// sum the tf and idf together and return a slice of floats
		tfIDFVec := tfIDF(sliceIntToFloat(tf), idf)

		fmt.Printf("Document %d TF: %v\n", i, tf)
		fmt.Printf("Document %d TF_IDF: %v\n", i, tfIDFVec)
	}

	// Example
	word := "dog"

	wordIdx, ok := vocab[word]
	if !ok {
		fmt.Printf("The word %s is not in the vocab", word)
	}

	// Find documents containing the word
	for i, doc := range tokens {
		tf := termFrequency(doc, vocab)
		if containsWord(sliceIntToFloat(tf), wordIdx) {
			fmt.Printf("Document %d contains %s\n", i, word)
		}
	}

	// Check how many times the word appears
	for i, doc := range tokens {
		tf := termFrequency(doc, vocab)
		count := tf[wordIdx]

		if count > 0 {
			fmt.Printf("Document %d: %s appears %d times\n", i, word, count)
		}
	}

	// Importance check across the corpus
	for i, doc := range tokens {
		tf := termFrequency(doc, vocab)
		tfIDFVec := tfIDF(sliceIntToFloat(tf), idf)

		score := tfIDFVec[wordIdx]
		if score > 0 {
			fmt.Printf("Document %d: TF-IDF(%s) = %.4f\n", i, word, score)
		}
	}

}

func containsWord(tf []float64, wordIdx int) bool {
	return tf[wordIdx] > 0
}

// termFrequency returns a slice of ints that represents how often each term shows up in a document
func termFrequency(doc []string, vocab map[string]int) []int {
	tf := make([]int, len(vocab))

	for _, word := range doc {
		if i, ok := vocab[word]; ok {
			tf[i]++
		}
	}

	return tf
}

func inverseDocumentFrequency(df []float64, numDocs int) []float64 {
	idf := make([]float64, len(df))

	for i, val := range df {
		// idf[i] = math.Log(float64(1+numDocs) / (1+val)) + 1 // smoothed IDF
		idf[i] = math.Log(float64(numDocs) / val)
	}

	return idf
}

func tfIDF(tf []float64, idf []float64) []float64 {
	result := make([]float64, len(tf))
	for i := range tf {
		result[i] = tf[i] * idf[i]
	}

	return result
}

// documentFrequency returns a slice representing how many times each word in the vocabulary shows up in all of the documents.
func documentFrequency(tokens [][]string, vocab map[string]int) []int {
	df := make([]int, len(vocab))

	for _, doc := range tokens {
		seen := make(map[int]bool)
		for _, word := range doc {
			if i, ok := vocab[word]; ok && !seen[i] {
				df[i]++
				seen[i] = true
			}
		}
	}

	return df
}

// tokenize returns tokens as space-delimited slices of strings of a corpus.
func tokenize(corpus []string) [][]string {
	var tokens [][]string
	for _, doc := range corpus {
		tokens = append(tokens, strings.Split(doc, " "))
	}

	return tokens
}

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

func sliceIntToFloat(s []int) []float64 {
	floatSlice := make([]float64, len(s))
	for i, v := range s {
		floatSlice[i] = float64(v)
	}

	return floatSlice
}
