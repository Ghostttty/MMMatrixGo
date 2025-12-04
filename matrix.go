package main

import (
	"math"
	"runtime"
	"sync"
)

type Matrix struct {
	X    uint32
	P    uint32
	Data []uint32
}

func CreateMatrix(X, P uint32) *Matrix {
	return &Matrix{X, P, make([]uint32, int(math.Pow(float64(X), float64(P))))}
}

// Multiplication выполняет матричное умножение с заданными параметрами lambda и mu
func (m *Matrix) Multiplication(lambda, mu uint32, other *Matrix) *Matrix {
	// Валидация входных данных
	if m == nil || other == nil {
		return nil
	}
	if m.X != other.X {
		panic("матрицы должны иметь одинаковую размерность X")
	}

	// Вычисление результирующей размерности
	resultP := (m.P - lambda - mu) + (other.P - lambda - mu) + lambda
	matrixResult := CreateMatrix(m.X, resultP)

	// Предварительные вычисления
	muPower := uint32(math.Pow(float64(m.X), float64(mu)))
	lastLHSIndex := int(m.P - 1)
	lastRHSIndex := int(lambda + mu - 1)

	// Инициализация индексных массивов
	indexLHS := make([]uint32, m.P)
	indexRHS := make([]uint32, other.P)
	indexMatrixResult := make([]uint32, matrixResult.P)

	for idx := range matrixResult.Data {
		var tempValue uint32

		if mu > 0 {
			// Многократное суммирование для mu > 0
			for sumIdx := uint32(0); sumIdx < muPower; sumIdx++ {
				tempValue += m.Data[calculateIndexFromArray(indexLHS, m.X)] *
					other.Data[calculateIndexFromArray(indexRHS, other.X)]

				if sumIdx+1 < muPower {
					incrementToIndexVector(indexLHS, lastLHSIndex, m.X)
					incrementToIndexVector(indexRHS, lastRHSIndex, m.X)
				}
			}
		} else {
			// Единичное умножение для mu = 0
			tempValue += m.Data[calculateIndexFromArray(indexLHS, m.X)] *
				other.Data[calculateIndexFromArray(indexRHS, other.X)]
		}

		matrixResult.Data[idx] = tempValue

		// Сброс индексов для следующей итерации
		filledZeroVector(indexLHS, lastLHSIndex, mu)
		filledZeroVector(indexRHS, lastRHSIndex, mu)

		// Обновление индексов результата
		if idx+1 < len(matrixResult.Data) {
			incrementIndexVector(indexMatrixResult, matrixResult.X)
			updateIndexMappings(indexMatrixResult, indexLHS, indexRHS, m.P, other.P, lambda, mu)
		}
	}

	return matrixResult
}

// ParallelMultiplication выполняет параллельное матричное умножение
func (m *Matrix) ParallelMultiplication(lambda, mu uint32, other *Matrix) *Matrix {
	if m == nil || other == nil {
		return nil
	}
	if m.X != other.X {
		panic("матрицы должны иметь одинаковую размерность X")
	}

	resultP := (m.P - lambda - mu) + (other.P - lambda - mu) + lambda
	size := int(math.Pow(float64(m.X), float64(resultP)))
	matrixResult := &Matrix{
		X:    m.X,
		P:    resultP,
		Data: make([]uint32, size),
	}

	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	chunkSize := (size + workers - 1) / workers

	// Используем пул для переиспользования массивов
	indexPool := &sync.Pool{
		New: func() interface{} {
			return &IndexBundle{
				lhs: make([]uint32, m.P),
				rhs: make([]uint32, other.P),
				res: make([]uint32, resultP),
			}
		},
	}

	for i := 0; i < size; i += chunkSize {
		end := i + chunkSize
		if end > size {
			end = size
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()

			muPower := uint32(math.Pow(float64(m.X), float64(mu)))
			lastLHSIndex := int(m.P - 1)
			lastRHSIndex := int(lambda + mu - 1)

			// Получаем bundle из пула
			bundle := indexPool.Get().(*IndexBundle)
			defer indexPool.Put(bundle)

			indexLHS := bundle.lhs
			indexRHS := bundle.rhs
			indexMatrixResult := bundle.res

			for idx := start; idx < end; idx++ {
				// Переиспользуем массивы, сбрасывая их перед использованием
				resetSlice(indexLHS)
				resetSlice(indexRHS)
				resetSlice(indexMatrixResult)

				// Вычисляем индекс
				fastCalculateIndexToArray(resultP, m.X, idx, indexMatrixResult)

				// Обновляем маппинги
				updateIndexMappings(indexMatrixResult, indexLHS, indexRHS, m.P, other.P, lambda, mu)

				var tempValue uint32

				if mu > 0 {
					// Создаем временные копии для инкремента
					tempLHS := make([]uint32, len(indexLHS))
					copy(tempLHS, indexLHS)
					tempRHS := make([]uint32, len(indexRHS))
					copy(tempRHS, indexRHS)

					for sumIdx := uint32(0); sumIdx < muPower; sumIdx++ {
						tempValue += m.Data[calculateIndexFromArray(tempLHS, m.X)] *
							other.Data[calculateIndexFromArray(tempRHS, other.X)]

						if sumIdx+1 < muPower {
							incrementToIndexVector(tempLHS, lastLHSIndex, m.X)
							incrementToIndexVector(tempRHS, lastRHSIndex, m.X)
						}
					}
				} else {
					tempValue += m.Data[calculateIndexFromArray(indexLHS, m.X)] *
						other.Data[calculateIndexFromArray(indexRHS, other.X)]
				}

				matrixResult.Data[idx] = tempValue
			}
		}(i, end)
	}

	wg.Wait()
	return matrixResult
}

// IndexBundle для группировки индексных массивов
type IndexBundle struct {
	lhs []uint32
	rhs []uint32
	res []uint32
}

// resetSlice обнуляет слайс
func resetSlice(slice []uint32) {
	for i := range slice {
		slice[i] = 0
	}
}

// updateIndexMappings обновляет сопоставления индексов между матрицами
func updateIndexMappings(indexMatrixResult, indexLHS, indexRHS []uint32,
	lhsP, rhsP uint32, lambda, mu uint32) {
	// Сопоставление для левой матрицы
	lhsSyncIdx := 0
	for i := 0; i < int(lhsP-mu); i++ {
		indexLHS[lhsSyncIdx] = indexMatrixResult[lhsSyncIdx]
		lhsSyncIdx++
	}

	// Сопоставление для правой матрицы (левая часть)
	rhsLeftSyncIdx := (len(indexMatrixResult) - int(lambda)) / 2
	rhsLeftIdx := 0
	for i := 0; i < int(lambda); i++ {
		indexRHS[rhsLeftIdx] = indexMatrixResult[rhsLeftSyncIdx]
		rhsLeftIdx++
		rhsLeftSyncIdx++
	}

	// Проверка на С значение
	if lambda+mu < uint32(len(indexMatrixResult)) {
		// Сопоставление для правой матрицы (правая часть)
		rhsRightSyncIdx := len(indexMatrixResult) - 1
		rhsRightIdx := int(rhsP - 1)
		for i := 0; i < int(rhsP-lambda-mu); i++ {
			indexRHS[rhsRightIdx] = indexMatrixResult[rhsRightSyncIdx]
			rhsRightIdx--
			rhsRightSyncIdx--
		}
	}
}

// incrementIndexVector увеличивает вектор индексов на 1 в заданной системе счисления
func incrementIndexVector(vec []uint32, base uint32) {
	for i := len(vec) - 1; i >= 0; i-- {
		if vec[i] == base-1 {
			vec[i] = 0
		} else {
			vec[i]++
			return
		}
	}
}
