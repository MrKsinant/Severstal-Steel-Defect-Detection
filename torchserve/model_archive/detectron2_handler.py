#!/usr/bin/env python3
# -*- coding: utf-8 -*-


################################################
# PROGRAMMER: Pierre-Antoine Ksinant           #
# DATE CREATED: 24/08/2020                     #
# REVISED DATE: -                              #
# PURPOSE: Custom handler for Detectron2 model #
################################################


##################
# Needed imports #
##################

import json, cv2, numpy, io, base64, detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import Visualizer


##################
# Custom service #
##################

class ModelHandler(object):
    """
    A base Model handler implementation for Detectron2
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.predictor = None
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model (called during model loading time)
        """
        print("Starting model initialization...")
        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
        cfg.MODEL.WEIGHTS = "detectron2_model.pth"
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.predictor = DefaultPredictor(cfg)
        self.initialized = True
        print("Model initialization done")

    def preprocess(self, batch):
        """
        Transform raw input into model input data
        """
        print("Starting raw input preprocessing...")
        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        model_input = []
        for request in batch:
            request_data = request.get("data")
            request_input = io.BytesIO(request_data)
            image = cv2.imdecode(numpy.fromstring(request_input.read(), numpy.uint8), 1)
            model_input.append(image)
        print("Raw input preprocessing done")
        return model_input

    def inference(self, model_input):
        """
        Internal inference method
        """
        print("Starting model inference...")
        model_output = []
        for image in model_input:
            outputs = self.predictor(image)
            model_inference = [image, outputs]
            model_output.append(model_inference)
        print("Model inference done")
        return model_output

    def postprocess(self, inference_output):
        """
        Return predict result in batch
        """
        print("Starting model inference output postprocessing...")
        results = []
        for model_inference in inference_output:
            image = model_inference[0]
            outputs = model_inference[1]
            predictions = outputs["instances"].to("cpu")
            severstal_metadata = Metadata()
            severstal_metadata.set(thing_classes=["Type 1", "Type 2", "Type 3", "Type 4"])
            visualizer_pred = Visualizer(image[:, :, ::-1], metadata=severstal_metadata, scale=0.5)
            image_pred = visualizer_pred.draw_instance_predictions(predictions)
            image_cv2 = cv2.cvtColor(image_pred.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            image_string = base64.b64encode(cv2.imencode(".jpg", image_cv2)[1].tobytes()).decode("utf-8")
            image_b64 = "data:image/jpg;base64," + image_string
            if predictions.has("pred_classes"):
                classes = predictions.pred_classes.numpy().tolist()
            else:
                classes = None
            if predictions.has("scores"):
                scores = predictions.scores.numpy().tolist()
            else:
                scores = None
            if predictions.has("pred_boxes"):
                boxes = predictions.pred_boxes.tensor.numpy().tolist()
            else:
                boxes = None
            if predictions.has("pred_masks"):
                #/!\ For an unknown reason (lack of memory, timeout...?), this doesn't work with TorchServe:
                #/!\ (it works perfectly in a Jupyter notebook!)
                #masks = predictions.pred_masks.numpy().tolist()
                masks = None
            else:
                masks = None
            result = {"data": image_b64, "classes": classes, "scores": scores, "boxes": boxes, "masks": masks}
            results.append(json.dumps(result))
        print("Model inference output postprocessing done")
        return results

    def handle(self, data, context):
        """
        Processing pipeline
        """
        print("Starting handing...")
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        results = self.postprocess(model_output)
        print("Handing done")
        return results

_service = ModelHandler()

def handle(data, context):
    """
    Handling function
    """
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None
    return _service.handle(data, context)