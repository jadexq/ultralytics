from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=1)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# results = model('/net/mulan/home/jadewang/robert/datasets/coco128/images/train2017/000000000009.npy')  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format

# check defacult config!!!!
