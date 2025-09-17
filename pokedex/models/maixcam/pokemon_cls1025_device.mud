type: classifier
cvimodel: /root/models/pokemon_cls1025_int8.cvimodel
input:
  format: rgb
  width: 224
  height: 224
preprocess:
  mean: [0.0, 0.0, 0.0]
  scale: [0.003922, 0.003922, 0.003922]
labels:
  num: 1025
  file: /root/models/classes.txt

