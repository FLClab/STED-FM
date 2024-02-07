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

try:
    import javabridge
    import bioformats
except ImportError:
    print("Bioformats does not seem to be installed on your machine...")
    print("Try running `pip install python-bioformats`")
    exit()

# Starts the java virtual machine
javabridge.start_vm(class_path=bioformats.JARS)

class MSRReader:
    """
    Creates a `MSRReader`. It will take some time to create the object

    :param logging_level: A `str` of the logging level to use {WARN, ERROR, OFF}

    :usage :
        with MSRReader() as msrreader:
            data = msrreader.read(file)
            image = data["STED_640"]
    """
    def __init__(self, logging_level="OFF"):

        rootLoggerName = javabridge.get_static_field("org/slf4j/Logger","ROOT_LOGGER_NAME", "Ljava/lang/String;")
        rootLogger = javabridge.static_call("org/slf4j/LoggerFactory","getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
        logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", logging_level, "Lch/qos/logback/classic/Level;")
        javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

    def read(self, msrfile):
        """
        Method that implements a `read` of the given `msrfile`

        :param msrfile: A file path to a `.msr` file

        :returns : A `dict` where each keys corresponds to a specific image
                   in the measurement file
        """
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

    def get_metadata(self, msrfile):
        """
        Method that returns a `dict` of the desired metadata of the image

        :param msrfile: A `str` of the file path to the `.msr` file

        :returns : A `dict` of the metadata
        """
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

    @staticmethod
    def get_info(metadata, startswith="get"):
        info = {}
        for func in filter(lambda func: func.startswith(startswith), dir(metadata)):
            key = "_".join(func.split("_")[1:])
            func = getattr(metadata, func)
            info[key] = func()
        return info

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
        "18-2022-10-26_SiR-Actin647_RachHD18_FixedDIV14_BCaMKIIA594-TauS488_cs1n1.msr"
    ]

    with MSRReader() as msrreader:
        for msrfile in msrfiles:
            data = msrreader.read(msrfile)
            metadata = msrreader.get_metadata(msrfile)
            for key, value in data.items():
                print(key, value.shape)
