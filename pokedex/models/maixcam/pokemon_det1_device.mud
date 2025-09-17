type: yolo11
model: /root/models/pokemon_det1_int8.cvimodel
input:
  format: rgb
  width: 256
  height: 256
preprocess:
  mean: [0.0, 0.0, 0.0]
  scale: [0.003922, 0.003922, 0.003922]
postprocess:
  conf_threshold: 0.35
  iou_threshold: 0.45
labels:
  num: 1
  names: ["pokemon"]

