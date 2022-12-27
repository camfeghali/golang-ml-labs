package main

import (
	"fmt"
	"os"
	"log"
	"strings"
	"strconv"
	"encoding/csv"
	"example.com/linreg"
)

func main() {
	records := readCsvFile("./dataset.csv")
	dataset := pick_dataset(records)
	normalized_dataset, means, sigmas := linreg.ZScoreNormalizeDataset(dataset)
	normalized_features := pick_features(normalized_dataset)
	targets := pick_targets(dataset)


	fmt.Printf("features: %f \n", dataset[0][:4])
	fmt.Printf("normalized_features: %f \n", normalized_features[0])

	model := &linreg.MultiVarLinearRegression{}
	model.Train(normalized_features, targets, 1.0e-1, 1000)

	w, b := model.GetParameters()
	fmt.Printf("w: %f\nb: %f \n", w, b)

	to_predict := []float64{1200.0, 3.0, 1.0, 40.0}
	normalized := linreg.ZScoreNormalize(to_predict, means, sigmas)

	fmt.Printf("X_train: %f \n", to_predict)
	fmt.Printf("X_norm: %f \n", normalized)

	prediction := model.Predict(normalized)
	fmt.Printf("prediction: %f \n", prediction)

}

func pick_features(dataset[][]float64) [][]float64 {

	features := [][]float64{}
	for _, row := range dataset {
		features = append(features, row[:len(row)-1])
	}
	return features
}

func pick_targets(dataset[][]float64) []float64 {
	targets := []float64{}
	row_length := len(dataset[0])
	for _, row := range dataset {
		value := row[row_length -1]
		targets = append(targets, value)
	}
	return targets
}

func pick_dataset(records [][]string) [][]float64 {
	dataset := [][]float64{}
	for _, row := range records[1:] {
		float_values := []float64{}
		for _, row_value := range row[1:len(row)] {
			float, _ := strconv.ParseFloat(strings.TrimSpace(row_value), 64)
			float_values = append(float_values, float)
		}
		dataset = append(dataset, float_values)
	}
	return dataset
}

func readCsvFile(filePath string) [][]string {
    f, err := os.Open(filePath)
    if err != nil {
        log.Fatal("Unable to read input file " + filePath, err)
    }
    defer f.Close()

    csvReader := csv.NewReader(f)
    records, err := csvReader.ReadAll()
    if err != nil {
        log.Fatal("Unable to parse file as CSV for " + filePath, err)
    }

    return records
}