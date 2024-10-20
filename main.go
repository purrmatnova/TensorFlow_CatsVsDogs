// Example how to use Cats vs Dogs ML model trained using Tensorflow on Python.
package main

import (
	"errors"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"io"
	"os"

	"github.com/nfnt/resize"
	tf "github.com/wamuir/graft/tensorflow"
)

const (
	imageDimension = 180
	outputOpName   = "StatefulPartitionedCall"
)

var serveTag = []string{"serve"} // Const.

var (
	ErrBadImageDimensions = errors.New("bad image dimensions")
	ErrCreateTensor       = errors.New("create tensor")
	ErrInvalidFormat      = errors.New("invalid image format")
	ErrInvalidOp          = errors.New("invalid or missing output operation")
	ErrLoadModel          = errors.New("could not load model")
	ErrRunSession         = errors.New("could not run the session")
	ErrWrongOperationName = errors.New("wrong operation name")
)

//nolint:gochecknoglobals // By design.
var (
	flagModelDir  string
	flagInputName string
)

func init() { //nolint:gochecknoinits // By design.
	flag.Usage = func() { //nolint:reassign // By design.
		_, _ = fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s [flags] file.jpg ...\n", os.Args[0])
		flag.PrintDefaults()
	}

	flag.StringVar(&flagModelDir, "model", "model", "directory with model")
	flag.StringVar(&flagInputName, "inputname", "serve_keras_tensor", "input operation name")
}

func main() {
	flag.Parse()

	if len(flag.Args()) == 0 {
		flag.Usage()
		os.Exit(0)
	}

	err := run(flagModelDir, flagInputName, flag.Args())
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func run(modelDir string, inputOpName string, paths []string) (err error) {
	model, err := loadModel(modelDir)
	if err != nil {
		return err
	}
	defer joinErr(&err, model.Session.Close)

	const catMaxProbability = 0.5
	const percent = 100
	for _, path := range paths {
		animal := "dog"
		probability, errPath := analyze(model, inputOpName, path)
		switch {
		case errPath != nil:
			err = errors.Join(err, errPath)
			continue
		case probability <= catMaxProbability:
			animal = "cat"
			probability = (1 - probability)
		}
		fmt.Printf("%s %5.2f%% %s\n", animal, probability*percent, path)
	}
	return err
}

func analyze(model *tf.SavedModel, inputOpName string, path string) (float32, error) {
	file, err := os.Open(path) //nolint:gosec // Safe in CLI.
	if err != nil {
		return 0, err
	}
	defer file.Close() //nolint:errcheck // False positive.

	probability, err := predict(model, inputOpName, file)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", path, err)
	}
	return probability, nil
}

func predict(model *tf.SavedModel, inputOpName string, imageReader io.Reader) (float32, error) {
	img, _, err := image.Decode(imageReader)
	if err != nil {
		return 0, fmt.Errorf("%w: %w", ErrInvalidFormat, err)
	}

	img = preprocessImage(img, imageDimension)

	inputTensor, err := tensorFromImage(img)
	if err != nil {
		return 0, err
	}

	// Создание карты для input операций
	op := model.Graph.Operation(inputOpName)
	if op == nil {
		return 0, fmt.Errorf("%w: input %s", ErrWrongOperationName, inputOpName)
	}
	inputs := map[tf.Output]*tf.Tensor{
		model.Graph.Operation(inputOpName).Output(0): inputTensor,
	}

	// Определяем output операции в зависимости от сигнатуры
	op2 := model.Graph.Operation(outputOpName)
	if op2 == nil {
		return 0, fmt.Errorf("%w: output %s", ErrWrongOperationName, outputOpName)
	}
	outputOp := model.Graph.Operation("StatefulPartitionedCall").Output(0)
	if outputOp == (tf.Output{}) {
		return 0, ErrInvalidOp
	}

	// Выполняем предсказание
	results, err := model.Session.Run(
		inputs,
		[]tf.Output{outputOp},
		nil,
	)
	if err != nil {
		return 0, fmt.Errorf("%w: %w", ErrRunSession, err)
	}

	probability := results[0].Value().([][]float32)[0][0]
	return probability, nil
}

func preprocessImage(img image.Image, dim uint) image.Image {
	return resize.Resize(dim, dim, img, resize.Lanczos3)
}

func tensorFromImage(img image.Image) (*tf.Tensor, error) {
	if img.Bounds().Dx() != imageDimension || img.Bounds().Dy() != imageDimension {
		return nil, fmt.Errorf("%w: %v", ErrBadImageDimensions, img.Bounds())
	}

	var imageData [imageDimension][imageDimension][3]float32
	for i := range imageData {
		for j := range imageData[i] {
			const bitsToByte = 24
			r, g, b, a := img.At(i, j).RGBA()
			imageData[i][j][0] = float32(r * a >> bitsToByte)
			imageData[i][j][1] = float32(g * a >> bitsToByte)
			imageData[i][j][2] = float32(b * a >> bitsToByte)
		}
	}

	tensor, err := tf.NewTensor([1][imageDimension][imageDimension][3]float32{imageData})
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrCreateTensor, err)
	}

	return tensor, nil
}

func loadModel(modelDir string) (*tf.SavedModel, error) {
	model, err := tf.LoadSavedModel(modelDir, serveTag, &tf.SessionOptions{})
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrLoadModel, err)
	}
	return model, nil
}

func joinErr(err *error, f func() error) { //nolint:gocritic // By design.
	*err = errors.Join(*err, f())
}
