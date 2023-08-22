import cv2 as cv
import numpy as np
import time
import os
import pdb
import matplotlib
import matplotlib.pyplot as plt

class NaiveBackgroundSubstraction:
    
    def __init__(self, n_frames, filename, output_path=None, time_interval=None):
        self.capture = cv.VideoCapture(filename)        
        if not self.capture.isOpened:
            print('Falla al abrir el archivo: ' + filename)
            exit(0)

        if n_frames:
            self.n_frames = n_frames
        else:
            self.n_frames = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT))

        self.fps = self.capture.get(cv.CAP_PROP_FPS)
        if time_interval:
            self.interval = time_interval * self.fps 
        else:
            self.interval = None
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
        frame_nums = int(self.capture.get(cv.CAP_PROP_FRAME_COUNT)) * np.random.uniform(size=self.n_frames)
        # pdb.set_trace()
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
                               (self.frame_width, self.frame_height))
        
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
            if self.interval and len(frame_queue) % self.interval == 0:
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
            
            output.write(cv.bitwise_and(frame, frame, mask=dframe))
            
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

class MOG2BackgroundSubstraction:
    
    def __init__(self, filename, output_path=None):
        self.capture = cv.VideoCapture(filename)    
        if not self.capture.isOpened:
            print('Falla al abrir el archivo: ' + filename)
            exit(0)
            
        self.backSub = cv.createBackgroundSubtractorMOG2()
        
        self.fps = self.capture.get(cv.CAP_PROP_FPS)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))
        
        ## create output folder
        output_path = output_path if output_path else "./output/"
        folder_exists = os.path.exists(output_path)
        if not folder_exists: 
            os.makedirs(output_path)
        self.output_path = output_path
    
    def apply(self, visualize=True, output_filename=None, erosion=False, dilation=False):
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        
        output_filename = output_filename if output_filename else "output.mp4"
        
        output = cv.VideoWriter(self.output_path + "/" + output_filename,
                               fourcc,
                               self.fps,
                               (self.frame_width, self.frame_height))
        
        # Reset frame number to 0
        self.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        
        # Corremos la sustraccion
        #------------------------
        start = time.time()      
        
        ret, frame = self.capture.read()
        
        while ret:
            # Leemos un frame
            if frame is None:
                break

            # Aplicamos la sustracción al frame leído
            #----------------------------------------
            # Cada frame se utiliza tanto para calcular la máscara de primer plano como para actualizar el fondo.
            # Si se desea cambiar la tasa de aprendizaje utilizada para actualizar el modelo de fondo, es posible
            # establecer una tasa de aprendizaje específica pasando un parámetro al método apply.
            fgMask = self.backSub.apply(frame)
            
            if erosion:
                kernel = np.ones((5,5),np.uint8)
                fgMask = cv.erode(fgMask, kernel, iterations = 1)
            
            if dilation:
                kernel = np.ones((5,5),np.uint8)
                fgMask = cv.dilate(fgMask, kernel, iterations = 1)
                
            output.write(cv.bitwise_and(frame, frame, mask=fgMask))
            
            ret, frame = self.capture.read()
            
            # mostrando la imagen
            if visualize:
                cv.imshow('FG Mask', fgMask)
                # Corremos hasta que termine o apriete escape
                keyboard = cv.waitKey(30)
                if keyboard == 'q' or keyboard == 27:
                    break


        elapsed = time.time()-start
        print('Tiempo de procesamiento {} segundos'.format(elapsed))
        cv.destroyAllWindows()
        
    def release_capture(self):
        self.capture.release()


output_path = "./output/"
input_path = "../videos_for_testing/"
black_list = ["getty_video2.mp4"]

for filename in sorted(os.listdir(input_path)):
    if filename.endswith(".mp4") and filename not in black_list:
        time_interval = 30
        output_filename = filename.replace(".mp4", "") + "_nbs.mp4"
        video_filename = input_path + filename

        print(video_filename)

        try:
            n_frames = 60
            nbs = NaiveBackgroundSubstraction(n_frames, video_filename, output_path, time_interval)
            nbs.apply(erosion=False, dilation=True, visualize=False, output_filename=output_filename)
            nbs.release_capture()
        except Exception as e:
            print("Revisar el file:", video_filename)
            # print(e)

# for i in range(1,8):
    
#     FILENAME = f'../videos_for_testing/video{i}.mp4'
#     time_interval = 5

#     try:
#         n_frames = 30
#         nbs = NaiveBackgroundSubstraction(n_frames, time_interval, FILENAME, output_path)
#         nbs.apply(erosion=False, dilation=True, visualize=False, output_filename=f"video{i}_nbs.mp4")
#         nbs.release_capture()
#     except:
#         continue

#     try:
#         mog2 = MOG2BackgroundSubstraction(FILENAME, output_path)
#         mog2.apply(visualize=False, output_filename=f"video{i}_mog.mp4", erosion=True, dilation=True)
#         mog2.release_capture()
#     except:
#         continue