import numpy as np

weights = np.load("converted_weights.npy", encoding='latin1').item()
print(len(weights))
keys = sorted(weights.keys())
print(keys)
for layer in keys:
	layer_weights = weights[layer]
	print(len((layer_weights)))
	weight_keys = sorted(layer_weights.keys())
	print(weight_keys)
	for item in weight_keys:
		if item=="biases":
			save_string = layer+"_b.npy"
		else:
			save_string = layer+"_W.npy"
		np.save(save_string, layer_weights[item])