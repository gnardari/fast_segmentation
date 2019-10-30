import tensorflow as tf
import sys
from tensorflow.core.framework import node_def_pb2

def remove_train_ops(graph_def):
    # Remove all ops that cannot run on the CPU
    removed = set()
    nodes = list(graph_def.node)
    for node in nodes:
        # if 'switch' in node.name.lower():
        if 'evaluation' in node.name.lower() or 'loss' in node.name.lower() or 'preds' in node.name.lower():
            graph_def.node.remove(node)
            removed.add(node.name)

    # Recursively remove ops depending on removed ops
    while removed:
        removed, prev_removed = set(), removed
        nodes = list(graph_def.node)
        for node in graph_def.node:
            if any(inp in prev_removed for inp in node.input):
                graph_def.node.remove(node)
                removed.add(node.name)

def make_placeholder(input_name, output_name):
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.bool, False, name=input_name)
    graph_def = graph.as_graph_def()
    graph_def.node[-1].name = output_name
    return graph_def.node

def switch_to_placeholder(graph_def):
    for n in graph_def.node:
        if n.op == 'Switch':
            input_name = n.input[0]
            output_name = n.name
            new_op = make_placeholder(input_name, output_name)
            graph_def.node.remove(n)
            graph_def.node.extend(new_op)
    return graph_def


# meta_path = 'models/fake_erfnet_nobn/snapshots_best/snapshot.chk.meta' # Your .meta file
# output_node_names = ['fres9/conv_b_1x3/Relu','d8/concat','up23/BiasAdd']    # Output nodes
# # output_node_names = ['preds/Argmax','preds/probs']    # Output nodes
# model = 'models/fake_erfnet_nobn/snapshots_best/snapshot.chk.data-00000-of-00001'

try:
    meta_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    out_path = sys.argv[3]
    output_nodes = []
    for out_node in sys.argv[4:]:
        output_nodes.append(out_node)
except Exception as e:
    print('Usage: python convert_to_pb.py path/to/model.chk.meta dir/of/checkpoint path/to/output.pb (outputNodeName)+')
    print(e)
    exit()

print('Metadata path: {}\nCkpt path: {}\nOutput model path: {}\nOutput Nodes: {}'.format(
    meta_path, ckpt_path, out_path, output_nodes))

with tf.Session() as sess:

    # Restore the graph
    # saver = tf.train.import_meta_graph(meta_path, input_map={'inputs/is_training': tf.convert_to_tensor(False)})
    saver = tf.train.import_meta_graph(meta_path, clear_devices=True)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint(ckpt_path))

    gdef = tf.get_default_graph().as_graph_def()
    clean_graph = tf.graph_util.remove_training_nodes(gdef)
    remove_train_ops(clean_graph)

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        clean_graph,
        output_nodes)

    # Save the frozen graph
    with open(out_path, 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())
