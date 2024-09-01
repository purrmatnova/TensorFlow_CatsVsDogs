package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"os"

	"github.com/nfnt/resize"
	"github.com/wamuir/graft/tensorflow"
)

func main() {
	// Укажем путь к модели и изображению
	modelDir := "./modelKeras/"
	imagePath := "cat.jpg"

	// Загрузим модель TensorFlow
	model, err := tensorflow.LoadSavedModel(modelDir, []string{"serve"}, nil)
	if err != nil {
		log.Fatalf("Ошибка загрузки модели: %v", err)
	}
	defer model.Session.Close()

	// Загрузим изображение
	imgFile, err := os.Open(imagePath)
	if err != nil {
		log.Fatalf("Не удалось открыть изображение: %v", err)
	}
	defer imgFile.Close()

	// Декодируем изображения JPEG
	img, err := jpeg.Decode(imgFile)
	if err != nil {
		log.Fatalf("Не удалось декодировать изображение: %v", err)
	}

	// Изменяем размер изображения до необходимого 256x256
	imgResized := resize.Resize(256, 256, img, resize.NearestNeighbor)

	// Конвертируем изображение в тензор формата []float32
	tensor, err := imageToTensor(imgResized)
	if err != nil {
		log.Fatalf("Ошибка преобразования изображения в тензор: %v", err)
	}

	// Выполняем предсказание
	output, err := model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			model.Graph.Operation("serving_default_input_1").Output(0): tensor,
		},
		[]tensorflow.Output{
			model.Graph.Operation("StatefulPartitionedCall").Output(0),
		},
		nil,
	)
	if err != nil {
		log.Fatalf("Ошибка выполнения модели: %v", err)
	}

	// Обрабатываем предсказание
	predictions := output[0].Value().([][]float32)
	fmt.Println("Результат предсказания: ", predictions)

	isCat := predictions[0][0] > 0.5
	if isCat {
		fmt.Println("На изображении кот.")
	} else {
		fmt.Println("На изображении не кот.")
	}
}

// Преобразуем изображение в тензор
func imageToTensor(img image.Image) (*tensorflow.Tensor, error) {
	var data [1][256][256][3]float32

	// Преобразуем изображение в массив float32
	for y := 0; y < 256; y++ {
		for x := 0; x < 256; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// Нормализуем значения пикселей (0-65535) до (0-1)
			data[0][y][x][0] = float32(r) / 65535.0
			data[0][y][x][1] = float32(g) / 65535.0
			data[0][y][x][2] = float32(b) / 65535.0
		}
	}

	// Преобразуем массив в тензор
	return tensorflow.NewTensor(data)
}
