
import os 
import json
import tarfile
import numpy 
import io

with tarfile.open("test.tar", "w") as tf:
    buffer = io.BytesIO()
    numpy.save(buffer, numpy.random.rand(128, 128).astype(numpy.float32))
    buffer.seek(0)

    info = tarfile.TarInfo(name="image1")
    info.size = len(buffer.getbuffer())
    tf.addfile(tarinfo=info, fileobj=buffer)

with tarfile.open("test.tar", "r") as tf:
    print(tf.getnames())
    buffer = io.BytesIO()
    buffer.write(tf.extractfile("image1").read())
    buffer.seek(0)
    img = numpy.load(buffer)
    print(img.shape)