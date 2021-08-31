import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os

pipeline_config_path = "D:/git/Metal_Corrosion_Classification/Detections/Tensorflow/workspace/models/ssd_mobnet2/pipeline.config"
labels = [{'name':'rust', 'id':1}]

config = config_util.get_configs_from_pipeline_file(pipeline_config_path)
print(config)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(pipeline_config_path, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = "D:/git/Metal_Corrosion_Classification/Detections/Tensorflow/workspace/pre-trained-models/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0"
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= "D:/git/Metal_Corrosion_Classification/Detections/Tensorflow/workspace/annotations/label_map.pbtxt"
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = ["D:/git/Metal_Corrosion_Classification/Detections/Tensorflow/workspace/annotations/train.record"]
pipeline_config.eval_input_reader[0].label_map_path = "D:/git/Metal_Corrosion_Classification/Detections/Tensorflow/workspace/annotations/label_map.pbtxt"
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = ["D:/git/Metal_Corrosion_Classification/Detections/Tensorflow/workspace/annotations/test.record"]

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(pipeline_config_path, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)

