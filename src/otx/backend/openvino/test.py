from otx.backends.openvino import OVEngine
from otx.backends.openvino.models import OVMulticlassClassificationModel

model = OVMulticlassClassificationModel()
ov_engine = OVEngine(model="/home/kprokofi/training_extensions/otx-workspace/20250327_171240/exported_model.xml",
                     data_root="/home/kprokofi/training_extensions/tests/assets/classification_dataset")
metrics = ov_engine.test()
print(metrics)
