import tensorflow as tf
import time
import numpy as np

frozen_graph_filename = '../../models/tree_loam_4w_filter_cos.pb'
with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

graph = tf.get_default_graph()
tf.import_graph_def(graph_def)

batch = graph.get_tensor_by_name('import/inputs/X:0')
prediction = graph.get_tensor_by_name('import/up23/BiasAdd:0')

times = []
with tf.Session() as sess:
    for _ in range(1000):
        start = time.time()
        values = sess.run(prediction, feed_dict={
            batch: np.random.uniform(size=(1,16,900,1)),
        })
        times.append(time.time() - start)

print('Avg time (ms): {}'.format(np.mean(times)*1000))
print(values.shape)
classes = np.argmax(values[0], axis=-1)
print(classes.shape)
print(np.bincount(classes.flatten()))
