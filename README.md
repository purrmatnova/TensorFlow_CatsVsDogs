# TensorFlow_CatsVsDogs

Пример реализации просто сверточной нейронной сети. 
Сам код модели находится в файле cats_vs_dogs_model.ipynb

Файл main.go содержит метод выполнения предсказания. 
Точность обученной сети - 0.8458. 

Перед запуском необходимо указать расположение CUDA

```console
 XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/cuda go run .
```
 
