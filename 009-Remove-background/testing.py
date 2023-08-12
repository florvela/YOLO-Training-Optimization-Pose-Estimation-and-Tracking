import cv2 as cv
import numpy as np
import time
import os
import pdb
import matplotlib
import matplotlib.pyplot as plt

class NaiveBackgroundSubstraction:
    
    def __init__(self, n_frames, time_interval, filename, output_path=None):
        self.capture = cv.VideoCapture(filename)        
        if not self.capture.isOpened:
            print('Falla al abrir el archivo: ' + filename)
            exit(0)
        
        self.n_frames = n_frames
        self.fps = self.capture.get(cv.CAP_PROP_FPS)
        self.interval = time_interval * self.fps 
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))
        self.set_background()
        
        ## create output folder
        output_path = output_path if output_path else "./output/"
        folder_exists = os.path.exists(output_path)
        if not folder_exists: 
            os.makedirs(output_path)
        self.output_path = output_path
    
    def get_video_background(self):
        frame_nums = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT)) * np.random.uniform(size=n_frames)
        frames = []
        for frame_num in frame_nums:
            self.capture.set(cv.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.capture.read()
            frames.append(frame)

        background = np.median(frames, axis=0).astype(dtype=np.uint8)
        return background
    
    def set_background(self, frame_queue=None):
        if frame_queue:
            frame_nums = len(frame_queue) * np.random.uniform(size=self.n_frames)
            frame_nums = frame_nums.astype(dtype=np.uint8)
            bg_frames = np.array(frame_queue)[frame_nums]
            self.background = np.median(bg_frames, axis=0).astype(dtype=np.uint8)
        else:
            self.background = self.get_video_background()
        
    def get_background(self):
        return self.background        
        
    def apply(self, erosion=False, dilation=False, visualize=True, output_filename=None):
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        
        output_filename = output_filename if output_filename else "output.mp4"
        
        output = cv.VideoWriter(self.output_path + "/" + output_filename,
                               fourcc,
                               self.fps,
                               (self.frame_width, self.frame_height),
                               False)
        
        # Reset frame number to 0
        self.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        
        start = time.time()

        # Convert background to grayscale
        gray_background = cv.cvtColor(self.background, cv.COLOR_BGR2GRAY)

        frame_queue = []
        
        ret, frame = self.capture.read()
        while ret:
            frame_queue.append(frame)

            # Si pasa el intervalo dado, re-calcular el fondo
            if len(frame_queue) % self.interval == 0:
                self.set_background(frame_queue)
                gray_background = cv.cvtColor(self.background, cv.COLOR_BGR2GRAY)
                frame_queue = []

            color_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # convertir el frame a escala de grises
            grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # calcular la diferencia del frame actual y el fondo
            dframe = cv.absdiff(grey_frame, gray_background)
            
            # binarizando
            th, dframe = cv.threshold(dframe, 30, 255, cv.THRESH_BINARY)
            
            if erosion:
                kernel = np.ones((5,5),np.uint8)
                dframe = cv.erode(dframe, kernel, iterations = 1)
            
            if dilation:
                kernel = np.ones((5,5),np.uint8)
                dframe = cv.dilate(dframe, kernel, iterations = 1)

            xxx = cv.bitwise_and(color_frame, color_frame, mask=dframe)

            # plt.imshow(xxx)
            # plt.show()
            # pdb.set_trace()
            
            print("grabando ando")                
            output.write(frame)
            
            # siguiente frame
            ret, frame = self.capture.read()
            
            # mostrando la imagen
            if visualize:
                cv.imshow('FG Mask', dframe)
                # Corremos hasta que termine o apriete ESC
                keyboard = cv.waitKey(30)
                if keyboard == 'q' or keyboard == 27:
                    break

        elapsed = time.time()-start
        print('Tiempo de procesamiento {} segundos'.format(elapsed))
        cv.destroyAllWindows()
    
    def release_capture(self):
        self.capture.release()

n_frames = 10
time_interval = 20 # in seconds
FILENAME = '../videos_for_testing/video6.mp4'
output_path = "./output/"

nbs = NaiveBackgroundSubstraction(n_frames, time_interval, FILENAME, output_path)

nbs.apply(erosion=True, dilation=True, visualize=False, output_filename="testing_mask.mp4")
nbs.release_capture()

# imprimiendo un frame del video
capture = cv.VideoCapture(output_path + "testing_mask.mp4")
capture.set(cv.CAP_PROP_POS_FRAMES, 540)
ret, frame = capture.read()
plt.imshow(frame)
capture.release()
plt.axis('off')
plt.show()