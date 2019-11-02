
import cv2
import argparse
import numpy as np
import glob
from tensorflow.python import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt

'''
The --mode flag is used to call either data, train, or test. Data mode is used
to save the optical flow frames from the video in order to be used in training.
Train mode loads the saved optical flow frame data and trains a Keras Sequential
model. Test mode loads the trained model and performs inference on the video,
saving the speed predictions to preds.npy. Use kalman_filter.py to smooth the
speed predictions.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--video', help='Video path')
parser.add_argument('--mode', default=None, help='data, train, or test, default to None')
args = parser.parse_args()

def create_model():
    model = Sequential()
    model.add(Convolution2D(32, 8,8, border_mode='same', subsample=(4,4), input_shape=(128,128,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 8,8, border_mode='same', subsample=(4,4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 4,4, border_mode='same', subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 2,2, border_mode='same', subsample=(1,1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def preprocess_frame(frame):
        # Crop dash of car and sky out of frame
        frame = frame[170:frame.shape[1]-290,:]

        # Adjust contrast and brightness of frame
        # Get histogram for gray scale frame
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        # # Get CDF of histogram
        # cdf = [float(hist[0])]
        # [cdf.append(cdf[-1] + float(hist[index])) for index in range(1, len(hist))]
        #
        # hist_percent = (cdf[-1]*0.125)
        # min_gray = np.argmin(np.abs(np.array(cdf) - hist_percent))
        # max_gray = np.argmin(np.abs(np.array(cdf) - cdf[-1] + hist_percent))
        # alpha = 255 / (max_gray - min_gray)
        # beta = -min_gray * alpha
        # frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return frame

def find_boxes(outs):
    # Return boxes of detected vehicles that are detected with > 50% confidence
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ['car',
                                                        'bicycle',
                                                        'motorcycle',
                                                        'bus',
                                                        'train',
                                                        'truck']:
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(detection[0] * width) - w / 2
                y = int(detection[1] * height) - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    return np.array(boxes)

def find_flow(frame,prev_frame,curr_frame,mask):
    # Use Farneback Optical Flow between successive frames
    flow = cv2.calcOpticalFlowFarneback(prev_frame,curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow image to have 3 channels
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    return rgb

def load_data():
    X, Y = [], []

    # Load flow images
    images = glob.glob('images/*')
    assert len(images) != 0, "No training data found"

    # Read and sort images by frame number
    images = sorted(images, key = lambda x: int(x.split('_')[-1].split('.')[0]))
    for image in images:
        im = cv2.imread(image)
        # Resize images to be 128 x 128
        im = cv2.resize(im,(128,128))
        X.append(im)
    X = np.array(X)

    # Read speed data
    Y = np.array([float(line.rstrip('\n')) for line in open('data/train.txt','r')][2:])
    Y = Y[:X.shape[0]]
    print(X.shape,Y.shape)

    return X, Y


def main():
    if args.mode == 'test':
        model = load_model('models/speed_model.h5')

    video = cv2.VideoCapture(args.video)

    global classes, width, height, net

    _, frame = video.read()
    frame = preprocess_frame(frame)
    width = frame.shape[1]
    height = frame.shape[0]
    scale = 0.00392

    classes = None
    with open('yolo/yolov3.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Initialize yolov3 detection model
    net = cv2.dnn.readNet('yolo/yolov3.weights', 'yolo/yolov3.cfg')

    prev_frame = None
    mask = np.zeros_like(frame)
    mask[..., 1] = 255

    # true_speed = np.array([float(line.rstrip('\n')) for line in open('data/train.txt','r')])[2:]
    # if args.mode == 'test':
    #     plt.ion()
    #     plt.show()

    all_speeds = []
    mse = []
    frame_count = 0

    if args.mode == 'data':
        assert len(glob.glob('images/*')) == 0, 'Data already found in images/ folder'

    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = preprocess_frame(frame)

        # Find boxes of vehicles in frame with yolo model
        blob = cv2.dnn.blobFromImage(frame, scale, (224,224), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        boxes = find_boxes(outs)

        # Change frame to grayscale
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None:
            prev_frame = curr_frame
            continue

        # Get optical flow from prev and curr frames
        flow = find_flow(frame,prev_frame,curr_frame,mask)

        # Black out boxes with vehicles or moving objects
        for box in boxes:
            cv2.rectangle(flow,(int(box[0]),int(box[1])),
                                (int(box[0]+box[2]),int(box[1]+box[3])),(0,0,0),-1)

        # Save flow to image folder
        if args.mode == 'data':
            cv2.imwrite('images/frame_'+str(frame_count)+'.jpg',flow)
            print('Processed frame {} out of {}'.format(frame_count,20400))

        frame_count += 1
        prev_frame = curr_frame.copy()

        # Use model to get speed prediction
        if args.mode == 'test':
            in_frame = np.expand_dims(cv2.resize(flow,(128,128)),axis=0)
            speed_pred = model.predict(in_frame).ravel()[0]

            # mse.append((speed_pred - true_speed[frame_count+1])**2)
            # curr_mse = round(np.mean(mse),3)
            all_speeds.append(speed_pred)
            cv2.putText(flow,'Predicted Speed: {} mph'.format(speed_pred),(20,50),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
            # cv2.putText(flow,'True Speed: {} mph'.format(true_speed[frame_count+1]),(20,80),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)
            # cv2.putText(flow,'Mean Squared Error: {}'.format(str(curr_mse)),(20,110),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2)

            # plt.close()
            # plt.plot(list(range(frame_count)),all_speeds,'r',label='Prediction')
            # plt.plot(list(range(frame_count)),true_speed[:frame_count],'b',label='True Speed')
            # plt.pause(0.01)

        cv2.imshow("Frame",frame)
        cv2.imshow("Flow",flow)

        if cv2.waitKey(1) == 27:
            break

    if args.mode == 'test':
        np.save('preds.npy',all_speeds)
    cv2.destroyAllWindows()


if args.mode == 'train':
    X, Y = load_data()
    model = create_model()
    model.fit(X, Y, batch_size=64, epochs=15, verbose=1, validation_split=0.2, shuffle=True)
    model.save('models/speed_model.h5')
else:
    main()






#
