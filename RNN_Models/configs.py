
model_configs = {
	"batch_size": 5,
	"num_epochs": 300,
	"rnn_layers_map": [
						# {"type": 'GRU', "unit": 100, "return_seq": True, "activation": 'relu'}, 
				   	 	{"type": 'GRU', "unit": 10, "return_seq": False, "activation": 'relu'}, 
				   		],
	"dense_layers_map": [
						{"unit": 10, "activation": 'linear'},
						# {"unit": 50, "activation": 'sigmoid'}, 
					 	{"unit": 1, "activation": 'linear'},
					 	],
	"optimizer": {"type": 'adam',
				  "default": False,
				  "lr": 0.001,
				  "decay": 0.0
				  },
	"loss": {"type": 'mean_squared_error'},
	"metrics": ['loss']
}


data_configs = {
	"test_ratio": 0.4,
	"time_step": 10,
	"look_forward": 1,
	"data_type": 'm2o', ## m2m(many to many) or m2o(many to one)
	"normalization": True
}
