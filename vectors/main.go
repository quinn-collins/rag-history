package main

import (
	"fmt"
	"math"
)

func main() {
	v1 := []float64{10, 20, 30, 40, 50}
	v2 := []float64{1, 2, 3, 4, 5}
	fmt.Println("v1 =", v1)
	fmt.Println("v2 =", v2)
	fmt.Println("v1 magnitude =", magnitude(v1))
	fmt.Println("v2 magnitude =", magnitude(v2))
	fmt.Println("v1+v2 =", add(v1, v2))
	fmt.Println("v1-v2 =", subtract(v1, v2))
	fmt.Println("v1*v2 =", multiply(v1, v2))
	fmt.Println("v1/v2 =", divide(v1, v2))
	fmt.Println("dot product similarity of v1 and v2 =", dotProduct(v1, v2))
	fmt.Println("v1 is orthoganal to v2 =", isOrthoganal(v1, v2))
	fmt.Println("cosine similarity of v1 and v2 =", cosineSimilarity(v1, v2))
	fmt.Println("classification =", classifyCosine(cosineSimilarity(v1, v2)))
}

func magnitude(vec []float64) float64 {
	var sum float64
	for _, v := range vec {
		sum += v * v
	}

	return math.Sqrt(sum)
}

// large the dot product the more the vectors are pointing the the same general direction
func dotProduct(a, b []float64) float64 {
	assertVectorsLengthEqual(a, b)

	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}

	return sum
}

// if the dot product of two vectors is 0 that means they are orthogonal to each other
func isOrthoganal(a, b []float64) bool {
	return dotProduct(a, b) == 0
}

// cosine of the angle is 1 means the vectors are identical in direction
// cosine of the angle is 0 means the vectors are orthogonal in direction
// normalizes the magnitude and defines only the direction
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

func classifyCosine(cos float64) string {
	switch {
	case cos == 1:
		return "Identical"
	case cos == 0:
		return "Orthogonal"
	case cos >= 0.8:
		return "Similar"
	case cos <= -0.8:
		return "Opposite"
	default:
		return "Unrelated"
	}
}

func add(a, b []float64) []float64 {
	assertVectorsLengthEqual(a, b)

	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}

	return result
}

func subtract(a, b []float64) []float64 {
	assertVectorsLengthEqual(a, b)

	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}

	return result
}

func multiply(a, b []float64) []float64 {
	assertVectorsLengthEqual(a, b)

	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}

	return result
}

func divide(a, b []float64) []float64 {
	assertVectorsLengthEqual(a, b)

	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] / b[i]
	}

	return result
}

func assertVectorsLengthEqual(a, b []float64) {
	if len(a) != len(b) {
		panic("vectors must have the same length")
	}
}
