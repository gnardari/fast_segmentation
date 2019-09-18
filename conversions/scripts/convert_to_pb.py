import tensorflow as tf
from tensorflow.core.framework import node_def_pb2


meta_path = 'models/fake_erfnet_nobn/snapshots_best/snapshot.chk.meta' # Your .meta file
output_node_names = ['fres9/conv_b_1x3/Relu','d8/concat','up23/BiasAdd']    # Output nodes
# output_node_names = ['preds/Argmax','preds/probs']    # Output nodes
model = 'models/fake_erfnet_nobn/snapshots_best/snapshot.chk.data-00000-of-00001'

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


# meta_graph_def = tf.MetaGraphDef()
# with open(meta_path, 'rb') as f:
#     meta_graph_def.MergeFromString(f.read())

# gdef = meta_graph_def.graph_def
# gdef = switch_to_placeholder(gdef)
# tf.graph_util.remove_training_nodes(tf.get_default_graph().as_graph_def())
# remove_train_ops(gdef)

# print([n.name for n in gdef.node])
# sess = tf.Session()
# saver = tf.train.import_meta_graph(gdef, clear_devices=True)
# saver.restore(sess, model)
# exit()
with tf.Session() as sess:

    # Restore the graph
    # saver = tf.train.import_meta_graph(meta_path)
    saver = tf.train.import_meta_graph(meta_path, input_map={'inputs/is_training': tf.convert_to_tensor(False)})
    # saver = tf.train.import_meta_graph(meta_graph_def, clear_devices=True)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('models/fake_erfnet_nobn/snapshots_best'))
    # saver.restore(sess,model)

    gdef = tf.get_default_graph().as_graph_def()
    print([n.name for n in gdef.node if 'BiasAdd' in n.name])
    # for n in gdef.node:
    #     if 'preds' in n.name:
    #         # gdef.node.remove(n)
    #         print(n.name)
        # if 'input' in n.name:
        #     print(n.name)

    # gdef = switch_to_placeholder(gdef)
    # remove_train_ops(gdef)
    clean_graph = tf.graph_util.remove_training_nodes(gdef)
    remove_train_ops(clean_graph)
    # print(([n.name for n in clean_graph.node]))

          # placeholder_node = node_def_pb2.NodeDef()
          # placeholder_node.op = "Const"
          # placeholder_node.name = n.name
          # placeholder_node.attr['value'] = False
          # gdef.node[nidx] = placeholder_node
    for nidx, n in enumerate(clean_graph.node):
        if n.op == 'Switch':
          print(n.name)

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        clean_graph,
        output_node_names)

    # Save the frozen graph
    with open('erfnet_nobn.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

