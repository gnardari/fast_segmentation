import tensorflow as tf
import numpy as np

frozen_graph_filename = '../../models/model.pb'
with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

graph = tf.get_default_graph()
tf.import_graph_def(graph_def)

print([n.name for n in tf.get_default_graph().as_graph_def().node])
exit()
ii = 'import/fres9/conv_b_1x3/Relu:0'
ij = 'import/d8/concat:0'
batch = graph.get_tensor_by_name('import/inputs/X:0')
prediction = graph.get_tensor_by_name('import/up23/BiasAdd:0')
firsti = graph.get_tensor_by_name(ii)
secondi = graph.get_tensor_by_name(ij)

with tf.Session() as sess:
    values = sess.run([firsti, secondi, prediction], feed_dict={
        batch: np.ones((1,256,256,1)),
    })
    print(values[0].shape)
    print(values[1].shape)
