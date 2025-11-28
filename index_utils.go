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

func incrementToIndexVector(vec []uint32, startIdx int, alignIncrement uint32) []uint32 {
	idx := startIdx
	for {
		if vec[idx] == alignIncrement-1 {
			vec[idx] = 0
			idx--
		} else {
			vec[idx]++
			return vec
		}
	}
}

func filledZeroVector(vec []uint32, startIdx int, mu uint32) []uint32 {
	idxCursor := startIdx
	for i := uint32(0); i < mu; i++ {
		vec[idxCursor] = 0
		if i+1 == mu {
			break
		}
		idxCursor--
	}
	return vec
}
