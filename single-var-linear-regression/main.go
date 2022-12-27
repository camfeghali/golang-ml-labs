package main

import (
	"os"
	"fmt"
	"log"
	"time"
	"strings"
	// "reflect"
	"strconv"
	"math/rand"
	"encoding/csv"
	"image/color"
	"example.com/linreg"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/plotter"
	// "gonum.org/v1/plot/plotutil"
)

func main() {
	model := &linreg.LinearRegression{}
	rand.Seed(time.Now().Unix())

	records := readCsvFile("./data.csv")
	data := records[1:]

	features := pickFeatures(data)
	scaled_features := linreg.ZScoreNormalize(features)
	targets := pickTargets(data)

	w, b, _, _ := model.Train(features, targets)

	fmt.Printf("w: %f \n", w)
	fmt.Printf("b: %f \n", b)
	// fmt.Printf("J_history: %f \n", J_history)
	// fmt.Printf("parameter_history: %f \n", parameter_history)

	fmt.Printf("features: %f \n", features)
	fmt.Printf("scaled_features: %f \n", scaled_features)

	model_points := generateModelPoints(features, model)

	points := MapToPoints(data)

	p := plot.New()
	p.Title.Text = "Linear Regression Model"
	p.X.Label.Text = ""
	p.Y.Label.Text = "Y"

	scatter_points, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter_points.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}

	// Make a line plotter and set its style.
	line, err := plotter.NewLine(model_points)
	if err != nil {
		panic(err)
	}
	line.LineStyle.Width = vg.Points(1)
	line.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	line.LineStyle.Color = color.RGBA{B: 255, A: 255}

	p.Add(scatter_points, line,)
	p.Legend.Add("Dataset", scatter_points)
	p.Legend.Add("Model", line)

	// Save the plot to a PNG file.
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "points.png"); err != nil {
		panic(err)
	}
}

func RandomFloats(count int, multiplier int) []float64 {
	floats := []float64{}
	multi := float64(multiplier)
	for i := 0; i <= count; i++ {
		floats = append(floats, multi * rand.Float64())
	}

	return floats
}

func makePoints(Xs, Ys []float64) plotter.XYs {
	var points plotter.XYs
	for idx, value := range Xs {
		points = append(points, makePoint(value, Ys[idx]))
	}
	return points
}

func makePoint(x, y float64) plotter.XY {
	return plotter.XY{
		X: x,
		Y: y,
	}
}

func randomPoints(n int) plotter.XYs {
	pts := make(plotter.XYs, n)
	for i := range pts {
		if i == 0 {
			pts[i].X = rand.Float64()
		} else {
			pts[i].X = pts[i-1].X + rand.Float64()
		}
		pts[i].Y = pts[i].X + 10*rand.Float64()
	}
	return pts
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

func MapToPoints(XYs [][]string) plotter.XYs {
	var points plotter.XYs
	for _, XY := range XYs {

		x, _ := strconv.ParseFloat(strings.TrimSpace(XY[0]), 64)
		y, _ := strconv.ParseFloat(strings.TrimSpace(XY[1]), 64)

		points = append(points, makePoint(x, y))
	}
	return points
}

func pickFeatures(rows [][]string) []float64 {
	var features []float64
	for _, row := range rows {

		feature, _ := strconv.ParseFloat(strings.TrimSpace(row[0]), 64)

		features = append(features, feature)
	}
	return features
}

func pickTargets(rows [][]string) []float64 {
	var targets []float64
	for _, row := range rows {

		target, _ := strconv.ParseFloat(strings.TrimSpace(row[1]), 64)

		targets = append(targets, target)
	}
	return targets
}

func generateModelPoints(features []float64, model *linreg.LinearRegression) plotter.XYs {
	var points plotter.XYs
	for _, feature_value := range features {

		prediction := model.Predict(feature_value)

		points = append(points, makePoint(feature_value, prediction))
	}
	return points
}