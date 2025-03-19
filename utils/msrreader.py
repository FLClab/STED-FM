"""
If running into import errors:
- Remove the try catch block to see the error --> probably "JVM not found"
- Make sure you have a working java version installed:
    In your terminal, run: $ java -version
        Output e.g.:
            java version "1.8.0_311"
            Java(TM) SE Runtime Environment (build 1.8.0_311-b11)
            Java HotSpot(TM) 64-Bit Server VM (build 25.311-b11, mixed mode)
- Make sure you have numpy installed (recommended to work in a virtual env):
    $ pip install numpy
- Uninstall and reinstall packages, starting with javabridge:
    $ pip uninstall javabridge
    $ pip uninstall python-bioformats
    $ pip install javabridge
    $ pip install python-bioformats
- If the error persists, you may need set the JAVA_HOME environment variable to the location where the
JVM is installed and try the pip install again.
    - To find the location of the JVM, run the following command in the terminal
        $ /usr/libexec/java_home
    - Copy paste the output and run:
        $ export $JAVA_HOME=<copied output>
    - Retry the pip uninstall/install

"""


import os
import numpy
import logging
import atexit
import threading
import queue
from threading import Thread

try:
    import javabridge
    import bioformats
except ImportError:
    print("Bioformats does not seem to be installed on your machine...")
    print("Try running `pip install python-bioformats`")
    exit()

log = logging.getLogger("JVM")
log.setLevel(logging.DEBUG)

# Starts the java virtual machine
# javabridge.start_vm(class_path=bioformats.JARS)

JAVABRIDGE_DEFAULT_LOG_LEVEL = "WARN"

class JavaBridgeException(Exception):
    pass

class _JBridgeThread(Thread):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

    def get_metadata(self, msrfile):
        self.task_queue.put(("get_metadata", msrfile))
        return self.result_queue.get()
    
    @staticmethod
    def get_info(metadata, startswith="get"):
        info = {}
        for func in filter(lambda func: func.startswith(startswith), dir(metadata)):
            key = "_".join(func.split("_")[1:])
            func = getattr(metadata, func)
            info[key] = func()
        return info

    def _get_metadata(self, msrfile):
        data = {}
        with bioformats.ImageReader(path=msrfile) as reader:
            metadata = bioformats.OMEXML(bioformats.get_omexml_metadata(path=reader.path))

            # Retreives the number of series
            series = metadata.get_image_count()
            for serie in range(series):
                image_metadata = metadata.image(serie)

                info = self.get_info(image_metadata)
                info.update(self.get_info(image_metadata.Pixels))
                data[image_metadata.get_Name()] = info
        return data

    def read(self, msrfile):
        self.task_queue.put(("read", msrfile))
        return self.result_queue.get()
    
    def _read(self, msrfile):
        data = {}
        with bioformats.ImageReader(path=msrfile) as reader:
            metadata = bioformats.OMEXML(bioformats.get_omexml_metadata(path=reader.path))

            # Retreives the number of series
            series = metadata.get_image_count()

            # We iterate over each serie
            rdr = reader.rdr
            for serie in range(series):
                rdr.setSeries(serie)
                X, Y, Z, T, C = rdr.getSizeX(), rdr.getSizeY(), rdr.getSizeZ(), rdr.getSizeT(), rdr.getSizeC()
                Zs = []
                for z in range(Z):
                    Ts = []
                    for t in range(T):
                        Cs = []
                        for c in range(C):
                            image = reader.read(z=z, t=t, c=c, series=serie, rescale=False)
                            Cs.append(image)
                        Ts.append(Cs)
                    Zs.append(Ts)

                # Avoids single axes in data
                image = numpy.array(Zs).squeeze()

                # Stores in data folder
                image_metadata = metadata.image(serie)
                data[image_metadata.get_Name()] = image 
        return data       

    def run(self) -> None:
        log.debug("Starting javabridge")
        javabridge.start_vm(class_path=bioformats.JARS)

        rootLoggerName = javabridge.get_static_field("org/slf4j/Logger","ROOT_LOGGER_NAME", "Ljava/lang/String;")
        rootLogger = javabridge.static_call("org/slf4j/LoggerFactory","getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
        logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", JAVABRIDGE_DEFAULT_LOG_LEVEL, "Lch/qos/logback/classic/Level;")
        javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)      

        while True:
            item = self.task_queue.get()
            if item is not None:
                command, msrfile = item
                method = getattr(self, f"_{command}")
                data = method(msrfile)
                self.result_queue.put(data)

    def kill(self):
        log.debug("Killing javabridge")
        javabridge.kill_vm()

class JVM:

    _jvm_thread = None
    _jvm_instance = None

    def __init__(self):
        JVM._jvm_thread = _JBridgeThread(daemon=True)
        JVM._jvm_thread.start()

    def __del__(self):
        JVM._jvm_thread.kill()

    @staticmethod
    def send(command, *args):
        if JVM._jvm_thread is not None:
            method = getattr(JVM._jvm_thread, command)
            return method(*args)

    @staticmethod
    def start():
        if JVM._jvm_instance is None:
            JVM._jvm_instance = JVM()

    @staticmethod
    def stop():
        if JVM._jvm_instance is not None:
            del JVM._jvm_instance
            JVM._jvm_instance = None

atexit.register(JVM.stop)
JVM.start()

class MSRReader:
    """
    Creates a `MSRReader`. It will take some time to create the object

    :param logging_level: A `str` of the logging level to use {WARN, ERROR, OFF}

    :usage :
        with MSRReader() as msrreader:
            data = msrreader.read(file)
            image = data["STED_640"]
    """
    def __init__(self):
        pass

    def read(self, msrfile):
        """
        Method that implements a `read` of the given `msrfile`

        :param msrfile: A file path to a `.msr` file

        :returns : A `dict` where each keys corresponds to a specific image
                   in the measurement file
        """
        return JVM.send("read", msrfile)

    def get_metadata(self, msrfile):
        """
        Method that returns a `dict` of the desired metadata of the image

        :param msrfile: A `str` of the file path to the `.msr` file

        :returns : A `dict` of the metadata
        """
        return JVM.send("get_metadata", msrfile)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()

if __name__ == "__main__":

    msrfiles = [
        # "BR0-Co-20mPFA_trs-shRNACamKIIBE_PSD95_GFP-MSMI_GAMSTAR580-Phalloidin_STAR635-02-01.msr",
        # "18-2022-10-26_SiR-Actin647_RachHD18_FixedDIV14_BCaMKIIA594-TauS488_cs1n1.msr"
        "/home-local2/projects/SSL/detection-data/confocal-actin/2024-01-12_Block_Adducin594-GPTau488-PH635_cs1n1.msr"
    ]

    with MSRReader() as msrreader:
        for msrfile in msrfiles:
            data = msrreader.read(msrfile)
            # metadata = msrreader.get_metadata(msrfile)
            # for key, value in data.items():
            #     print(key, value.shape)
