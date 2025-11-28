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

func CreateMatrix(X, P uint32) Matrix {
	return Matrix{X, P, make([]uint32, int(math.Pow(float64(X), float64(P))))}
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
	matrixResult := &Matrix{
		X:    m.X,
		P:    resultP,
		Data: make([]uint32, int(math.Pow(float64(m.X), float64(resultP)))),
	}

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
					indexLHS = incrementToIndexVector(indexLHS, lastLHSIndex, m.X)
					indexRHS = incrementToIndexVector(indexRHS, lastRHSIndex, m.X)
				}
			}
		} else {
			// Единичное умножение для mu = 0
			tempValue += m.Data[calculateIndexFromArray(indexLHS, m.X)] *
				other.Data[calculateIndexFromArray(indexRHS, other.X)]
		}

		matrixResult.Data[idx] = tempValue

		// Сброс индексов для следующей итерации
		indexLHS = filledZeroVector(indexLHS, lastLHSIndex, mu)
		indexRHS = filledZeroVector(indexRHS, lastRHSIndex, mu)

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
	// Валидация входных данных
	if m == nil || other == nil {
		return nil
	}
	if m.X != other.X {
		panic("матрицы должны иметь одинаковую размерность X")
	}

	// Вычисление результирующей размерности
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

	// Создание канала для задач
	taskChan := make(chan [2]int, workers)

	// Запуск воркеров
	for i := 0; i < workers; i++ {
		go m.multiplicationWorker(lambda, mu, other, matrixResult, taskChan, &wg)
	}

	// Распределение задач
	for i := 0; i < size; i += chunkSize {
		end := i + chunkSize
		if end > size {
			end = size
		}
		wg.Add(1)
		taskChan <- [2]int{i, end}
	}

	close(taskChan)
	wg.Wait()
	return matrixResult
}

// multiplicationWorker обрабатывает диапазон индексов для параллельного умножения
func (m *Matrix) multiplicationWorker(lambda, mu uint32, other, result *Matrix, taskChan chan [2]int, wg *sync.WaitGroup) {
	muPower := uint32(math.Pow(float64(m.X), float64(mu)))
	lastLHSIndex := int(m.P - 1)
	lastRHSIndex := int(lambda + mu - 1)

	for task := range taskChan {
		start, end := task[0], task[1]
		for idx := start; idx < end; idx++ {
			// Инициализация индексных массивов для текущей позиции
			indexLHS := make([]uint32, m.P)
			indexRHS := make([]uint32, other.P)
			indexMatrixResult := calculateIndexToArray(result.P, result.X, idx)

			// Обновление сопоставлений индексов
			updateIndexMappings(indexMatrixResult, indexLHS, indexRHS, m.P, other.P, lambda, mu)

			var tempValue uint32

			if mu > 0 {
				for sumIdx := uint32(0); sumIdx < muPower; sumIdx++ {
					tempValue += m.Data[calculateIndexFromArray(indexLHS, m.X)] *
						other.Data[calculateIndexFromArray(indexRHS, other.X)]

					if sumIdx+1 < muPower {
						indexLHS = incrementToIndexVector(indexLHS, lastLHSIndex, m.X)
						indexRHS = incrementToIndexVector(indexRHS, lastRHSIndex, m.X)
					}
				}
			} else {
				tempValue += m.Data[calculateIndexFromArray(indexLHS, m.X)] *
					other.Data[calculateIndexFromArray(indexRHS, other.X)]
			}

			result.Data[idx] = tempValue
		}
		wg.Done()
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
		for i := 0; i < int(lhsP-lambda-mu); i++ {
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
