package main

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {

	model, err := tf.LoadSavedModel("mnistmodel", []string{"serve"}, nil)

	if err != nil {
		fmt.Printf("Error loading saved model: %s\n", err.Error())
		return
	}

	defer model.Session.Close()

	tensor, terr := dummyInputTensor(28 * 28) // replace this with your own data
	if terr != nil {
		fmt.Printf("Error creating input tensor: %s\n", terr.Error())
		return
	}

	result, runErr := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("imageinput").Output(0): tensor,
		},
		[]tf.Output{
			model.Graph.Operation("infer").Output(0),
		},
		nil,
	)

	if runErr != nil {
		fmt.Printf("Error running the session with input, err: %s\n", runErr.Error())
		return
	}

	fmt.Printf("Most likely number in input is %v \n", result[0].Value())

}

func dummyInputTensor(size int) (*tf.Tensor, error) {

	imageData := [][]float32{make([]float32, size)}
	return tf.NewTensor(imageData)
}
