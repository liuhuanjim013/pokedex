[basic]
type = cvimodel
model = /root/models/pokemon_det1_int8.cvimodel

[extra]
model_type = yolo11
type = detector
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
conf_threshold = 0.35
iou_threshold = 0.45
labels = pokemon

