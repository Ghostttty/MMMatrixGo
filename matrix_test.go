package main

import (
	"fmt"
	"math"
	"testing"
)

// TestCalculateIndexFromArray тестирует корректность вычисления индекса
func TestCalculateIndexFromArray(t *testing.T) {
	tests := []struct {
		name     string
		array    []uint32
		x        uint32
		expected int
	}{
		{
			name:     "2D array index calculation",
			array:    []uint32{1, 2},
			x:        3,
			expected: 5, // 1*3^1 + 2*3^0 = 3 + 2 = 5
		},
		{
			name:     "3D array index calculation",
			array:    []uint32{1, 0, 2},
			x:        3,
			expected: 11, // 1*3^2 + 0*3^1 + 2*3^0 = 9 + 0 + 2 = 11
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateIndexFromArray(tt.array, tt.x)
			if result != tt.expected {
				t.Errorf("calculateIndexFromArray(%v, %d) = %d, expected %d",
					tt.array, tt.x, result, tt.expected)
			}
		})
	}
}

// TestCalculateIndexToArray тестирует обратное преобразование
func TestCalculateIndexToArray(t *testing.T) {
	tests := []struct {
		name     string
		p        uint32
		x        uint32
		index    int
		expected []uint32
	}{
		{
			name:     "Index to 2D array",
			p:        2,
			x:        3,
			index:    5,
			expected: []uint32{1, 2}, // 5 = 1*3 + 2
		},
		{
			name:     "Index to 3D array",
			p:        3,
			x:        3,
			index:    11,
			expected: []uint32{1, 0, 2}, // 11 = 1*9 + 0*3 + 2
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateIndexToArray(tt.p, tt.x, tt.index)
			if len(result) != len(tt.expected) {
				t.Errorf("Length mismatch: got %d, expected %d", len(result), len(tt.expected))
				return
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("Position %d: got %d, expected %d", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

// TestIndexConversionConsistency тестирует согласованность прямого и обратного преобразования
func TestIndexConversionConsistency(t *testing.T) {
	testCases := []struct {
		p     uint32
		x     uint32
		index int
	}{
		{p: 2, x: 3, index: 5},
		{p: 3, x: 3, index: 11},
		{p: 2, x: 4, index: 7},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("p=%d_x=%d_index=%d", tc.p, tc.x, tc.index), func(t *testing.T) {
			// Преобразуем индекс в массив
			array := calculateIndexToArray(tc.p, tc.x, tc.index)

			// Преобразуем массив обратно в индекс
			calculatedIndex := calculateIndexFromArray(array, tc.x)

			// Должны получить исходный индекс
			if calculatedIndex != tc.index {
				t.Errorf("Index conversion inconsistent: original=%d, calculated=%d, array=%v",
					tc.index, calculatedIndex, array)
			}
		})
	}
}

// TestMatrixMultiplicationConsistency тестирует, что последовательная и параллельная версии дают одинаковый результат
func TestMatrixMultiplicationConsistency(t *testing.T) {
	// Создаем тестовые матрицы
	lhs := CreateMatrix(3, 2)
	lhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

	rhs := CreateMatrix(3, 2)
	rhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

	// Тестируем различные параметры
	params := []struct {
		lambda uint32
		mu     uint32
	}{
		{lambda: 1, mu: 1},
		{lambda: 0, mu: 0},
		{lambda: 1, mu: 0},
		{lambda: 0, mu: 1},
	}

	for _, param := range params {
		testName := fmt.Sprintf("lambda_%d_mu_%d", param.lambda, param.mu)
		t.Run(testName, func(t *testing.T) {
			// Вычисляем результат последовательным методом
			sequential := lhs.Multiplication(param.lambda, param.mu, rhs)

			// Вычисляем результат параллельным методом
			parallel := lhs.ParallelMultiplication(param.lambda, param.mu, rhs)

			// Проверяем, что результаты идентичны
			compareMatrices(t, sequential, parallel)
		})
	}
}

// TestMatrixProperties тестирует свойства матричных операций
func TestMatrixProperties(t *testing.T) {
	// Создаем тестовую матрицу
	matrix := CreateMatrix(2, 2)
	matrix.Data = []uint32{1, 2, 3, 4}

	// Создаем нулевую матрицу
	zeroMatrix := CreateMatrix(2, 2)
	matrix.Data = []uint32{0, 0, 0, 0}

	t.Run("Multiplication with zero matrix", func(t *testing.T) {
		// Умножение на нулевую матрицу должно дать матрицу определенной размерности
		result := matrix.Multiplication(1, 0, zeroMatrix)

		// Проверяем размерность результата
		expectedSize := int(math.Pow(float64(result.X), float64(result.P)))
		if len(result.Data) != expectedSize {
			t.Errorf("Result size incorrect: got %d, expected %d", len(result.Data), expectedSize)
		}
	})

	t.Run("Zero matrix multiplication", func(t *testing.T) {
		// Умножение нулевой матрицы на другую матрицу
		result := zeroMatrix.Multiplication(1, 0, matrix)

		// Проверяем размерность
		expectedSize := int(math.Pow(float64(result.X), float64(result.P)))
		if len(result.Data) != expectedSize {
			t.Errorf("Result size incorrect: got %d, expected %d", len(result.Data), expectedSize)
		}
	})
}

// TestEdgeCases тестирует граничные случаи
func TestEdgeCases(t *testing.T) {
	t.Run("Single element matrices", func(t *testing.T) {
		a := CreateMatrix(2, 1)
		a.Data = []uint32{5, 3}

		b := CreateMatrix(2, 1)
		b.Data = []uint32{3, 5}

		result := a.Multiplication(0, 0, b)

		// Проверяем размерность результата
		expectedSize := int(math.Pow(float64(result.X), float64(result.P)))
		if len(result.Data) != expectedSize {
			t.Errorf("Single element result size incorrect: got %d, expected %d",
				len(result.Data), expectedSize)
		}
	})

	t.Run("Different sizes", func(t *testing.T) {
		a := CreateMatrix(3, 2)
		a.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

		b := CreateMatrix(3, 1)
		b.Data = []uint32{1, 2, 3}

		// Это должно работать, если логика умножения поддерживает разные размерности
		result := a.Multiplication(0, 1, b)

		expectedSize := int(math.Pow(float64(result.X), float64(result.P)))
		if len(result.Data) != expectedSize {
			t.Errorf("Different sizes result incorrect: got %d, expected %d",
				len(result.Data), expectedSize)
		}
	})
}

// TestResultMatrixDimensions тестирует корректность размерностей результирующей матрицы
func TestResultMatrixDimensions(t *testing.T) {
	testCases := []struct {
		pA, pB     uint32
		lambda, mu uint32
		expectedP  uint32
	}{
		{pA: 2, pB: 2, lambda: 1, mu: 1, expectedP: 1}, // (2-1-1) + (2-1-1) + 1 = 1
		{pA: 3, pB: 3, lambda: 1, mu: 1, expectedP: 3}, // (3-1-1) + (3-1-1) + 1 = 3
		{pA: 4, pB: 4, lambda: 2, mu: 1, expectedP: 4}, // (4-2-1) + (4-2-1) + 2 = 4
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("pA=%d_pB=%d_lambda=%d_mu=%d", tc.pA, tc.pB, tc.lambda, tc.mu), func(t *testing.T) {
			a := CreateMatrix(2, tc.pA)
			b := CreateMatrix(2, tc.pB)

			result := a.Multiplication(tc.lambda, tc.mu, b)

			if result.P != tc.expectedP {
				t.Errorf("Incorrect result dimension: got %d, expected %d", result.P, tc.expectedP)
			}

			expectedSize := int(math.Pow(float64(result.X), float64(result.P)))
			if len(result.Data) != expectedSize {
				t.Errorf("Incorrect result size: got %d, expected %d", len(result.Data), expectedSize)
			}
		})
	}
}

// TestKnownMultiplicaton тестирует известные случаи умножения
func TestKnownMultiplicaton(t *testing.T) {
	t.Helper()

	t.Run("3*3 X 3*3 (1,1)", func(t *testing.T) {
		lhs := CreateMatrix(3, 2)
		lhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

		rhs := CreateMatrix(3, 2)
		rhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

		res := []uint32{14, 77, 194}
		res1 := lhs.Multiplication(1, 1, rhs)
		res2 := lhs.ParallelMultiplication(1, 1, rhs)

		compareMatrices(t, res1, res2)

		if len(res1.Data) != len(res) {
			t.Errorf("Result length mismatch: got %d, expected %d", len(res1.Data), len(res))
			return
		}

		for i := range res1.Data {
			if res1.Data[i] != res[i] {
				t.Errorf("Result mismatch at index %d: got %d, expected %d", i, res1.Data[i], res[i])
				return
			}
		}
	})

	t.Run("3*3 X 3*1 (0,1)", func(t *testing.T) {
		lhs := CreateMatrix(3, 2)
		lhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

		rhs := CreateMatrix(3, 1)
		rhs.Data = []uint32{1, 2, 3}

		res := []uint32{14, 32, 50}
		res1 := lhs.Multiplication(0, 1, rhs)
		res2 := lhs.ParallelMultiplication(0, 1, rhs)

		compareMatrices(t, res1, res2)

		if len(res1.Data) != len(res) {
			t.Errorf("Result length mismatch: got %d, expected %d", len(res1.Data), len(res))
			return
		}

		for i := range res1.Data {
			if res1.Data[i] != res[i] {
				t.Errorf("Result mismatch at index %d: got %d, expected %d", i, res1.Data[i], res[i])
				return
			}
		}
	})

	t.Run("3*3 X 3*3 (0,1)", func(t *testing.T) {
		lhs := CreateMatrix(3, 2)
		lhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

		rhs := CreateMatrix(3, 2)
		rhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

		res := []uint32{30, 36, 42, 66, 81, 96, 102, 126, 150}
		res1 := lhs.Multiplication(0, 1, rhs)
		res2 := lhs.ParallelMultiplication(0, 1, rhs)

		compareMatrices(t, res1, res2)

		if len(res1.Data) != len(res) {
			t.Errorf("Result length mismatch: got %d, expected %d", len(res1.Data), len(res))
			return
		}

		for i := range res1.Data {
			if res1.Data[i] != res[i] {
				t.Errorf("Result mismatch at index %d: got %d, expected %d", i, res1.Data[i], res[i])
				return
			}
		}
	})

	t.Run("3*3 X 3*3 (1,0)", func(t *testing.T) {
		lhs := CreateMatrix(3, 2)
		lhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

		rhs := CreateMatrix(3, 2)
		rhs.Data = []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9}

		res := []uint32{1, 2, 3, 8, 10, 12, 21, 24, 27, 4, 8, 12, 20, 25, 30, 42, 48, 54, 7, 14, 21, 32, 40, 48, 63, 72, 81}
		res1 := lhs.Multiplication(1, 0, rhs)
		res2 := lhs.ParallelMultiplication(1, 0, rhs)

		compareMatrices(t, res1, res2)

		if len(res1.Data) != len(res) {
			t.Errorf("Result length mismatch: got %d, expected %d", len(res1.Data), len(res))
			return
		}

		for i := range res1.Data {
			if res1.Data[i] != res[i] {
				t.Errorf("Result mismatch at index %d: got %d, expected %d", i, res1.Data[i], res[i])
				return
			}
		}
	})
}

// Benchmark тесты для измерения производительности
func BenchmarkMultiplications(b *testing.B) {
	// Создаем матрицы для бенчмарков
	lhs := CreateMatrix(3, 3)

	// Заполняем данными
	for i := range lhs.Data {
		lhs.Data[i] = uint32(i % 5)
	}

	rhs := CreateMatrix(3, 3)

	for i := range rhs.Data {
		rhs.Data[i] = uint32((i + 2) % 5)
	}

	b.ResetTimer()

	b.Run("Sequential", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			lhs.Multiplication(1, 1, rhs)
		}
	})

	b.Run("Parallel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			lhs.ParallelMultiplication(1, 1, rhs)
		}
	})
}

// Benchmark тесты на разные типы умножения
func BenchmarkMultiplicationsWithDetails(b *testing.B) {
	// Создаем тестовые данные один раз
	lhs := CreateMatrix(3, 3)
	for i := range lhs.Data {
		lhs.Data[i] = uint32(i % 5)
	}

	rhs := CreateMatrix(3, 3)
	for i := range rhs.Data {
		rhs.Data[i] = uint32((i + 2) % 5)
	}

	// Тестируем разные конфигурации
	configs := []struct {
		name       string
		lambda, mu uint32
	}{
		{"Lambda1_Mu1", 1, 1},
		{"Lambda0_Mu0", 0, 0},
		{"Lambda1_Mu0", 1, 0},
		{"Lambda0_Mu1", 0, 1},
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			// Сбрасываем таймер после настройки
			b.ReportAllocs()
			b.ResetTimer()

			// Запускаем бенчмарк
			for i := 0; i < b.N; i++ {
				lhs.Multiplication(config.lambda, config.mu, rhs)
			}
		})
	}
}

// Parallel Benchmark тесты на разные типы умножения
func BenchmarkParallelMultiplicationsWithDetails(b *testing.B) {
	// Создаем тестовые данные один раз
	lhs := CreateMatrix(3, 3)
	for i := range lhs.Data {
		lhs.Data[i] = uint32(i % 5)
	}

	rhs := CreateMatrix(3, 3)
	for i := range rhs.Data {
		rhs.Data[i] = uint32((i + 2) % 5)
	}

	// Тестируем разные конфигурации
	configs := []struct {
		name       string
		lambda, mu uint32
	}{
		{"Lambda1_Mu1", 1, 1},
		{"Lambda0_Mu0", 0, 0},
		{"Lambda1_Mu0", 1, 0},
		{"Lambda0_Mu1", 0, 1},
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			// Сбрасываем таймер после настройки
			b.ReportAllocs()
			b.ResetTimer()

			// Запускаем бенчмарк
			for i := 0; i < b.N; i++ {
				lhs.ParallelMultiplication(config.lambda, config.mu, rhs)
			}
		})
	}
}

// Benchmark тесты для измерения производительности
func BenchmarkBigMultiplications(b *testing.B) {
	// Создаем матрицы для бенчмарков
	lhs := CreateMatrix(10, 6)

	// Заполняем данными
	for i := range lhs.Data {
		lhs.Data[i] = uint32(i % 5)
	}

	rhs := CreateMatrix(10, 6)

	for i := range rhs.Data {
		rhs.Data[i] = uint32((i + 2) % 5)
	}

	b.ResetTimer()

	configs := []struct {
		name       string
		lambda, mu uint32
	}{
		{"Lambda4_Mu0", 4, 0},
		{"Lambda2_Mu2", 2, 2},
		{"Lambda3_Mu1", 3, 1},
		{"Lambda1_Mu3", 1, 3},
	}

	for _, config := range configs {
		b.Run(config.name+"_Parallel", func(b *testing.B) {
			// Сбрасываем таймер после настройки
			b.ReportAllocs()
			b.ResetTimer()

			// Запускаем бенчмарк
			for i := 0; i < b.N; i++ {
				lhs.ParallelMultiplication(config.lambda, config.mu, rhs)
			}
		})
		b.Run(config.name+"_Sequential", func(b *testing.B) {
			// Сбрасываем таймер после настройки
			b.ReportAllocs()
			b.ResetTimer()

			// Запускаем бенчмарк
			for i := 0; i < b.N; i++ {
				lhs.Multiplication(config.lambda, config.mu, rhs)
			}
		})
	}
}

func generatePairs(lhsP, rhsP uint32) [][2]uint32 {
	var pairs [][2]uint32

	for l := uint32(0); l <= min(lhsP, rhsP); l++ {
		for m := uint32(0); m <= min(lhsP, rhsP); m++ {
			if l == m && m == 0 {
				continue
			}

			if l+m > min(lhsP, rhsP) {
				continue
			}

			if lhsP == rhsP && m == rhsP {
				continue
			}

			pairs = append(pairs, [2]uint32{l, m})
		}
	}

	return pairs
}

func BenchmarkFullTest(b *testing.B) {
	X := uint32(10)

	for lhsP := uint32(2); lhsP < 4; lhsP++ {

		lhs := CreateMatrix(X, lhsP)

		// Заполняем данными
		for i := range lhs.Data {
			lhs.Data[i] = uint32(i % 5)
		}

		for rhsP := uint32(1); rhsP <= lhsP; rhsP++ {

			rhs := CreateMatrix(X, rhsP)

			for i := range rhs.Data {
				rhs.Data[i] = uint32((i + 2) % 5)
			}

			b.ResetTimer()

			for _, pair := range generatePairs(lhsP, rhsP) {
				lambda := pair[0]
				mu := pair[1]

				b.Run(fmt.Sprintf("Parallel lhs P=%d rhs P=%d lambda=%d mu=%d", lhsP, rhsP, lambda, mu), func(b *testing.B) {
					// Сбрасываем таймер после настройки
					b.ReportAllocs()
					b.ResetTimer()

					// Запускаем бенчмарк
					for i := 0; i < b.N; i++ {
						lhs.ParallelMultiplication(lambda, mu, rhs)
					}
				})
				b.Run(fmt.Sprintf("Sequential lhs P=%d rhs P=%d lambda=%d mu=%d", lhsP, rhsP, lambda, mu), func(b *testing.B) {
					// Сбрасываем таймер после настройки
					b.ReportAllocs()
					b.ResetTimer()

					// Запускаем бенчмарк
					for i := 0; i < b.N; i++ {
						lhs.Multiplication(lambda, mu, rhs)
					}
				})
			}

		}
	}
}

// Вспомогательные функции

// compareMatrices сравнивает две матрицы на идентичность
func compareMatrices(t *testing.T, a, b *Matrix) {
	t.Helper()

	if a == nil || b == nil {
		t.Error("One of matrices is nil")
		return
	}

	if a.X != b.X {
		t.Errorf("X mismatch: %d vs %d", a.X, b.X)
	}

	if a.P != b.P {
		t.Errorf("P mismatch: %d vs %d", a.P, b.P)
	}

	if len(a.Data) != len(b.Data) {
		t.Errorf("Data length mismatch: %d vs %d", len(a.Data), len(b.Data))
		return
	}

	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Errorf("Data mismatch at index %d: %d vs %d", i, a.Data[i], b.Data[i])
			return
		}
	}
}
