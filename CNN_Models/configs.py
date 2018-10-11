
model_configs = {
	"batch_size": 256,
	"num_epochs": 1,
	"cnn_layers_map": [
						{"type": 'Conv2D', "filters": 32, "kernal_size": (3,3), "activation": 'relu', "input": True},
						{"type": 'Conv2D', "filters": 64, "kernal_size": (3,3), "activation": 'relu', "input": False},
						{"type": 'MaxPooling2D', "pool_size": (2,2)},
						{"type": 'Dropout', "rate": 0.25},
						{"type": 'Flatten'},
						{"type": 'Dense', "units": 128, "activation": 'relu'},
						{"type": 'Dropout', "rate": 0.5},
						{"type": 'Dense', "units": 10, "activation": 'softmax'}
						],
	"optimizer": {"type": 'adadelta',
				  "default": True,
				  "lr": 1.0,
				  "decay": 0.0
				  },
	"loss": {"type": 'categorical_crossentropy'},
	"metrics": ['accuracy']
}


data_configs = {

}
