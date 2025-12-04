package main

import "math"

func calculateIndexFromArray(arrayIndex []uint32, x uint32) int {
	var resultIndex uint32
	n := len(arrayIndex)
	for idx := 0; idx < n-1; idx++ {
		power := uint32(math.Pow(float64(x), float64(n-idx-1)))
		resultIndex += arrayIndex[idx] * power
	}
	resultIndex += arrayIndex[n-1]
	return int(resultIndex)
}

func calculateIndexToArray(p uint32, x uint32, index int) []uint32 {
	resultVector := make([]uint32, p)
	temp := uint32(index)

	for idx := range resultVector {
		power := uint32(math.Pow(float64(x), float64(p-uint32(idx)-1)))
		resultVector[idx] = min(temp/power, x-1)
		temp -= resultVector[idx] * power
	}

	return resultVector
}

func fastCalculateIndexToArray(p uint32, x uint32, index int, result []uint32) {
	temp := uint32(index)
	for i := len(result) - 1; i >= 0; i-- {
		result[i] = temp % x
		temp /= x
	}
}

func incrementToIndexVector(vec []uint32, lastIndex int, base uint32) {
	for i := lastIndex; i >= 0; i-- {
		if vec[i] == base-1 {
			vec[i] = 0
		} else {
			vec[i]++
			return
		}
	}
}

func filledZeroVector(vec []uint32, startIdx int, mu uint32) {
	idxCursor := startIdx
	for i := uint32(0); i < mu; i++ {
		vec[idxCursor] = 0
		if i+1 == mu {
			break
		}
		idxCursor--
	}
}
