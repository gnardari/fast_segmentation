import tensorflow as tf
import time
import cv2
import numpy as np

frozen_graph_filename = '../../models/kittiroad/kittiroad.pb'
with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

graph = tf.get_default_graph()
tf.import_graph_def(graph_def)

batch = graph.get_tensor_by_name('import/inputs/X:0')
prediction = graph.get_tensor_by_name('import/up23/BiasAdd:0')

im = cv2.imread('../../data/input2.png')
im = cv2.resize(im, (1200,368))
print(im.shape)

times = []
with tf.Session() as sess:
    for _ in range(1):
        start = time.time()
        values = sess.run(prediction, feed_dict={
            batch: [im],
        })
        times.append(time.time() - start)

print('Avg time (ms): {}'.format(np.mean(times)*1000))
print('Output dims: {}'.format(values.shape))

out = np.argmax(values[0], axis=-1)
print(out.shape)
print(np.bincount(out.flatten()))

cv2.imwrite('../../data/out.png', out*255.)
