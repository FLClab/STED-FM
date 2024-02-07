
import numpy
import tarfile
import io

OUTPATH = "/home-local2/projects/FLCDataset/dataset.tar"
with tarfile.open(OUTPATH, "r") as tf:
    
    members = tf.getmembers()
    for member in members:
        buffer = io.BytesIO()
        buffer.write(tf.extractfile(member.name).read())
        buffer.seek(0)
        data = numpy.load(buffer, allow_pickle=True)
        print(data["image"].shape, data["metadata"])