import tensorflow as tf
import uff

pb_file_path = "../../models/erfnet_nobn.pb"
with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
with open(pb_file_path, "rb") as f:
    output_graph_def.ParseFromString(f.read())

uff_model = uff.from_tensorflow(graphdef=output_graph_def,
output_nodes=["up23/BiasAdd"],
output_filename="../../models/model.uff",
text=True)
