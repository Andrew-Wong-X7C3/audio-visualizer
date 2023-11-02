# ====================================================================================

import sys

import numpy as np
import scipy as sp
import pyaudiowpatch as pyaudio

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# ====================================================================================

# pyaudio constants
FORMAT = pyaudio.paInt16
CHUNK = 1024
CHANNELS = 1

# signal processing constants
DOWNSAMPLE_RATIO = 16
SAMPLING_WINDOW = 16

POLY_DEGREE = 13

FILTER_ORDER = 5
FILTER_FREQUENCY = 50
SPLINE_DEGREE = 3
SPLINE_SMOOTHING = None

# graph constants
REFRESH_RATE = 0
PLOY_LINE_REFRESH_RATIO = 5
COLOR_RESOLUTION = 512

MIN_X = 2.5
MAX_X = 4.3
MIN_Y = 0
MAX_Y = 25

# ====================================================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # init audio listener
        self.device = None
        self.stream = None
        self.setup()

        # create plot
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.graphWidget.setBackground('black')
        self.graphWidget.setLogMode(True, False)
        self.graphWidget.setXRange(MIN_X, MAX_X)
        self.graphWidget.setYRange(MIN_Y, MAX_Y)
        self.graphWidget.getPlotItem().hideAxis('bottom')
        self.graphWidget.getPlotItem().hideAxis('left')

        # init graph data
        self.counter = SAMPLING_WINDOW
        self.buffer = np.zeros((SAMPLING_WINDOW, CHUNK))

        self.fft_x = np.fft.rfftfreq(CHUNK * SAMPLING_WINDOW, d=1/self.device["defaultSampleRate"])
        self.downsample_x = np.mean(self.fft_x[:-1].reshape(-1, DOWNSAMPLE_RATIO), axis=1)
        self.fft_y = np.zeros(len(self.fft_x))

        self.poly_x = np.linspace(0, self.fft_x[-1], CHUNK)
        self.poly_y = np.zeros(CHUNK)

        # create plot color
        poly_color = pg.colormap.get('gray', source='matplotlib')
        poly_pen = poly_color.getPen(span=(MIN_Y, MAX_Y))
        self.poly_line_1 = self.graphWidget.plot(self.poly_x, self.poly_y, pen=poly_pen)
        self.poly_line_2 = self.graphWidget.plot(self.poly_x, self.poly_y, pen=poly_pen)

        fft_color = pg.colormap.get('viridis', source='matplotlib')
        fft_pen = fft_color.getPen(span=(MIN_Y, MAX_Y / 2))
        self.fft_line = self.graphWidget.plot(self.fft_x, self.fft_y, pen=fft_pen)

        # create update loop
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(REFRESH_RATE * 1e3))
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()


    def keyPressEvent(self, e):

        # controls key press events
        if e.key() == QtCore.Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()

        if e.key() == QtCore.Qt.Key_F:
            self.showFullScreen()


    def setup(self):

        # Get default WASAPI info
        p = pyaudio.PyAudio()
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        except OSError:
            print("Looks like WASAPI is not available on the system. Exiting...")
            exit()

        # get loopback device
        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print("Default loopback output device not found.\n\nRun `python -m pyaudiowpatch` to check available devices.\nExiting...\n")
                exit()

        # open target device stream
        self.device = default_speakers
        self.stream = p.open(
            format=FORMAT,
            channels=self.device['maxInputChannels'],
            rate=int(self.device['defaultSampleRate']),
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=default_speakers['index']
        )
    

    def update_plot_data(self):

        # get current frame data and resample to 1 channel
        frame = self.stream.read(CHUNK)
        data = np.frombuffer(frame, dtype=np.int16)
        data = np.mean(data.reshape(-1, self.device['maxInputChannels']), axis=1)

        # update frame buffer and flatten
        index = int((SAMPLING_WINDOW / 4) + (self.counter % (SAMPLING_WINDOW / 2)))
        self.buffer[index] = data
        self.counter += 1
        self.fft_y = self.buffer.flatten()

       # shift up, and normalize, zero-pad, and poly fit
        self.poly_y = data.copy()
        self.poly_y[:int(CHUNK / 4)] = np.zeros(int(CHUNK / 4))
        self.poly_y[-int(CHUNK / 4):] = np.zeros(int(CHUNK / 4))
        self.poly_y = (MAX_Y / 2) * self.poly_y / (max(self.poly_y) + 0.01)
        self.poly_y += np.tile((MAX_Y - MIN_Y) / 2, len(self.poly_y))
        
        sigma = np.ones(CHUNK)
        sigma[:int(CHUNK / 16)] = 1e1
        sigma[int(-CHUNK / 16):] = 1e1
        z = np.polyfit(self.poly_x, self.poly_y, POLY_DEGREE, w=sigma)
        self.poly_y = np.polyval(z, self.poly_x)

        # calculate fft and convert to magnitudes
        self.fft_y = np.fft.rfft(self.fft_y)
        self.fft_y = np.abs(self.fft_y) / len(self.fft_y)

        # perform B-spline fit
        t, c, k = sp.interpolate.splrep(self.fft_x, self.fft_y, s=SPLINE_SMOOTHING, k=SPLINE_DEGREE)
        spline = sp.interpolate.BSpline(t, c, k, extrapolate=False)
        self.fft_y = spline(self.fft_x)

        # down-sample and scale amplitudes by volume / frequency
        self.fft_y = np.mean(self.fft_y[:-1].reshape(-1, DOWNSAMPLE_RATIO), axis=1)
        if max(self.fft_y) > 1:
            self.fft_y = np.multiply(self.fft_y, np.linspace(1e-1, 1e1, len(self.fft_y)))

        # update plot
        if self.counter % PLOY_LINE_REFRESH_RATIO == 0:
            self.poly_line_1.setData(self.poly_x, self.poly_y)
            self.poly_line_2.setData(self.poly_x[::-1], self.poly_y)

        self.fft_line.setData(self.downsample_x, self.fft_y)

# ====================================================================================

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
w.showFullScreen()
sys.exit(app.exec_())