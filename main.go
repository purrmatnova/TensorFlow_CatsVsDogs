package main

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"log"
	"os"

	"github.com/nfnt/resize"
	tensorflow "github.com/wamuir/graft/tensorflow"
)

const dim = 180

// Предобработка изображения
func preprocessImage(img image.Image) (*tensorflow.Tensor, error) {
	resized := resize.Resize(dim, dim, img, resize.Lanczos3)
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, resized, nil); err != nil {
		return nil, fmt.Errorf("failed to encode image: %v", err)
	}

	imageBytes := buf.Bytes()
	var imageData [dim][dim][3]float32
	index := 0
	for i := range imageData {
		for j := range imageData[i] {
			for k := range imageData[i][j] {
				if index < len(imageBytes) {
					imageData[i][j][k] = float32(imageBytes[index]) / 255.0
					index++
				}
			}
		}
	}

	tensor, err := tensorflow.NewTensor([1][dim][dim][3]float32{imageData})
	if err != nil {
		return nil, fmt.Errorf("failed to create tensor: %v", err)
	}

	return tensor, nil
}

// Загрузка модели
func loadModel(modelDir string) (*tensorflow.SavedModel, error) {
	model, err := tensorflow.LoadSavedModel(modelDir, []string{"serve"}, &tensorflow.SessionOptions{})
	if err != nil {
		return nil, fmt.Errorf("could not load model: %v", err)
	}
	return model, nil
}

var (
	ErrInvalidFormat = errors.New("invalid image format")
	ErrInvalidOp     = errors.New("invalid or missing output operation")
	ErrRunSession    = errors.New("could not run the session")
)

func predict(file io.Reader) (float32, error) {
	img, _, err := image.Decode(file)
	if err != nil {
		return 0, fmt.Errorf("%w: %w", ErrInvalidFormat, err)
	}
	tensor, err := preprocessImage(img)
	if err != nil {
		return 0, err
	}
	model, err := loadModel("modelDir")
	if err != nil {
		return 0, err
	}
	defer model.Session.Close()

	// Создание карты для input операций
	op := model.Graph.Operation("serve_keras_tensor")
	if op == nil {
		return 0, errors.New("operation not found by name")
	}
	inputs := map[tensorflow.Output]*tensorflow.Tensor{
		model.Graph.Operation("serve_keras_tensor").Output(0): tensor,
	}

	// Определяем output операции в зависимости от сигнатуры
	op2 := model.Graph.Operation("StatefulPartitionedCall")
	if op2 == nil {
		return 0, errors.New("output operation not found by name")
	}
	outputOp := model.Graph.Operation("StatefulPartitionedCall").Output(0)
	if outputOp == (tensorflow.Output{}) {
		return 0, ErrInvalidOp
	}

	// Выполняем предсказание
	results, err := model.Session.Run(
		inputs,
		[]tensorflow.Output{outputOp},
		nil,
	)
	if err != nil {
		return 0, fmt.Errorf("%w: %w", ErrRunSession, err)
	}

	probability := results[0].Value().([][]float32)[0][0]
	return probability, nil
}

func main() {
	file, err := os.Open("cat.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	probability, err := predict(file)
	if err != nil {
		log.Fatal(err)
	}
	if probability <= 0.5 {
		fmt.Printf("Скорее всего, на картинке изображен котик. Вероятность - %.2f%%\n", (1-probability)*100)
	} else {
		fmt.Printf("Скорее всего, на картинке изображена собака. Вероятность - %.2f%%\n", probability*100)
	}

	fmt.Println(probability)
}
