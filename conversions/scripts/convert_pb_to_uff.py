import tensorflow as tf
import uff
import sys

# pb_file_path = "../../models/erfnet_nobn.pb"
# uff_model = uff.from_tensorflow(graphdef=output_graph_def,
# output_nodes=["up23/BiasAdd"],
# output_filename="../../models/model.uff",
# text=True)

try:
    pb_file_path = sys.argv[1]
    output_filename = sys.argv[2]
    output_nodes = []
    for out_node in sys.argv[3:]:
        output_nodes.append(out_node)
except Exception as e:
    print('Usage: python convert_pb_to_uff.py path/to/model.pb path/to/output.uff (outputNodeName)+')
    print(e)
    exit()

print('Input file: {}\nOutput File: {}\nOutput Nodes: {}'.format(
    pb_file_path, output_filename, output_nodes))

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
with open(pb_file_path, "rb") as f:
    output_graph_def.ParseFromString(f.read())

uff.from_tensorflow(graphdef=output_graph_def,
                    output_nodes=output_nodes,
                    output_filename=output_filename,
                    text=True)
