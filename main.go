package main

import (
	"fmt"
	"log"
)

func main() {
	fmt.Println("Starting Matrix Program...")

	lhs := CreateMatrix(3,2)
	lhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	
	rhs := CreateMatrix(3,2)
	rhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

	// Тестируем оба метода
	res1 := lhs.Multiplication(0, 1, rhs)
	res2 := lhs.ParallelMultiplication(0, 1, rhs)

	fmt.Printf("Sequential result: %v\n", res1.Data)
	fmt.Printf("Parallel result: %v\n", res2.Data)

	// Проверяем что результаты идентичны
	match := true
	for i := range res1.Data {
		if res1.Data[i] != res2.Data[i] {
			match = false
			break
		}
	}

	if match {
		fmt.Println("✓ Sequential and parallel results match!")
	} else {
		fmt.Println("✗ Sequential and parallel results differ!")
	}
}

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}
