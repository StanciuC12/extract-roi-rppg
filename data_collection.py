import datetime
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from collections import deque


class FrameProcessor:
    def __init__(self, transform, roi_scale_factor=0.5):

        self.haarcascade = "haarcascade_frontalface_alt2.xml"
        self.LBFmodel = "lbfmodel.yaml"
        self.detector = cv2.CascadeClassifier(self.haarcascade)
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(self.LBFmodel)
        self.roi_scale_factor = roi_scale_factor
        self.square_sizes = [64, 128]
        self.transform = transform

    def extract_roi(self, img, save_roi_image=False):

        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(image_gray)
        if len(faces) != 1:
            print('.')  #Face not detected
            return None
        _, landmarks = self.landmark_detector.fit(image_gray, faces)
        image_with_annotations = img.copy()
        cropped_images = {}

        for landmark in landmarks:
            if save_roi_image:
                for idx, (x, y) in enumerate(landmark[0]):
                    cv2.circle(image_with_annotations, (int(x), int(y)), 1, (255, 255, 255), 1)
                    cv2.putText(image_with_annotations, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                                cv2.LINE_AA)

            # point_2 = landmark[0][1]  # point 2
            # point_29 = landmark[0][30]  # point 29
            # point_14 = landmark[0][15]  # point 14

            point_2 = landmark[0][2]  # point 2
            point_29 = landmark[0][30]  # point 29
            point_14 = landmark[0][14]  # point 14

            square_size = int(np.min([np.abs(point_2 - point_29)[0], np.abs(point_14 - point_29)[0]]) * self.roi_scale_factor)
            if square_size > 100:
                square_size = 128
            elif square_size > 50:
                square_size = 64
            elif square_size > 25:
                square_size = 32
            else:
                square_size = 32
                # print('Square too small')
                # return None

            left_square_center = np.mean([point_2, point_29], axis=0).astype(int)
            right_square_center = np.mean([point_29, point_14], axis=0).astype(int)
            diff1 = np.abs(point_2 - point_29)
            diff2 = np.abs(point_29 - point_14)

            if diff1[0] > square_size:  # don't do left cheek
                left_square_top_left = (
                left_square_center[0] - square_size // 2, left_square_center[1] - square_size // 2)
                left_square_bottom_right = (
                left_square_center[0] + square_size // 2, left_square_center[1] + square_size // 2)
                left_cheek = img[left_square_top_left[1]:left_square_bottom_right[1],
                             left_square_top_left[0]:left_square_bottom_right[0]]
                cropped_images['left_cheek'] = left_cheek
                if save_roi_image:
                    cv2.rectangle(image_with_annotations, left_square_top_left, left_square_bottom_right, (255, 0, 0), 2)

            if diff2[0] > square_size:
                right_square_top_left = (
                right_square_center[0] - square_size // 2, right_square_center[1] - square_size // 2)
                right_square_bottom_right = (
                    right_square_center[0] + square_size // 2, right_square_center[1] + square_size // 2)
                right_cheek = img[right_square_top_left[1]:right_square_bottom_right[1],
                              right_square_top_left[0]:right_square_bottom_right[0]]
                cropped_images['right_cheek'] = right_cheek
                if save_roi_image:
                    cv2.rectangle(image_with_annotations, right_square_top_left, right_square_bottom_right, (255, 0, 0), 2)

            upper_points = landmark[0][17:27]
            topmost_point = min(upper_points, key=lambda point: point[1])
            square_size = square_size * 2
            forehead_square_center = [upper_points[:, 0].mean(), topmost_point[1]]
            forehead_square_top_left = (
                int(forehead_square_center[0] - square_size // 2), int(forehead_square_center[1] - square_size))
            forehead_square_bottom_right = (
                int(forehead_square_center[0] + square_size // 2), int(forehead_square_center[1]))
            forehead = img[forehead_square_top_left[1]:forehead_square_bottom_right[1],
                       forehead_square_top_left[0]:forehead_square_bottom_right[0]]
            cropped_images['forehead'] = forehead
            if save_roi_image:
                cv2.rectangle(image_with_annotations, forehead_square_top_left, forehead_square_bottom_right, (255, 0, 0), 2)
                cropped_images['img_with_annotations'] = image_with_annotations

        return cropped_images

    def display_video_with_roi(self, capture):
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            cropped_images = self.extract_roi(frame, save_roi_image=True)
            if cropped_images is None:
                continue

            cv2.imshow('Original Frame', frame)

            if 'img_with_annotations' in cropped_images:
                cv2.imshow('Annotated Frame', cropped_images['img_with_annotations'])

            for roi in ['right_cheek', 'left_cheek', 'forehead']:
                if roi in cropped_images:
                    cv2.imshow(f'ROI {roi}', cropped_images[roi])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def get_data_from_capture(self, capture, num_frames=150, save_adr=None):

        frames_list = deque(maxlen=3 * num_frames)  # 3 ROI * nr frames
        i = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            cropped_images = self.extract_roi(frame, save_roi_image=True)
            if cropped_images is None:
                continue

            for roi in ['right_cheek', 'left_cheek', 'forehead']:
                if roi in cropped_images:
                    roi_img = cropped_images[roi]
                    roi_img_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                    roi_img_pil = Image.fromarray(roi_img_rgb)
                    frame_tensor = self.transform(roi_img_pil)
                    frames_list.append(frame_tensor)

            if 'img_with_annotations' in cropped_images:
                annotated_frame = cropped_images['img_with_annotations']
                cv2.imshow('Annotated Frame', annotated_frame)

            cv2.imshow('Original Frame', frame)

            for idx, roi in enumerate(['right_cheek', 'left_cheek', 'forehead']):
                if roi in cropped_images:
                    cv2.imshow(f'ROI {roi}', cropped_images[roi])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if len(frames_list) >= 3 * num_frames and i % 30 == 0:  #once per second
                #print(len(frames_list))
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                frames_tensor = torch.stack(list(frames_list))
                frames_tensor = frames_tensor.view(1, 3 * 3, num_frames, 64, 64)

                if save_adr is not None:
                    output_file = f'out_t_{timestamp}.pkl'
                    with open(os.path.join(save_adr, output_file), 'wb') as f:
                        pickle.dump(frames_tensor, f)

                yield frames_tensor

            i += 1

        capture.release()
        cv2.destroyAllWindows()

    def get_spo2_df(self, data_folder, spo2_csvs, save_adr):

        df = pd.DataFrame(columns=['data_adr', 'spo2'])

        spo2_dfs = []
        for csv in spo2_csvs:

            spo2_dfs.append(pd.read_csv(csv))

        spo2_df = pd.concat(spo2_dfs, ignore_index=True)
        spo2_df['Datetime'] = pd.to_datetime(spo2_df['DATE'] + ' ' + spo2_df['TIME'])

        i = 0
        for file in os.listdir(data_folder):

            time = file.split('_t_')[-1].split('.')[0]
            format_str = '%Y-%m-%d_%H-%M-%S'
            datetime_obj = datetime.datetime.strptime(time, format_str)

            diff = (spo2_df['Datetime'] - datetime_obj).abs()
            is_under_2_seconds = any(diff < pd.Timedelta(seconds=2))

            if is_under_2_seconds:
                closest_index = diff.idxmin()
                spo2_value = spo2_df.loc[closest_index, 'SPO2']

                df.loc[i] = [os.path.join(data_folder, file), spo2_value]
                i += 1
            else:
                print('No value found for video', file)

        df.to_excel(save_adr)



if __name__ == "__main__":

    # Parameters ######################################################################################################
    capture = True
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))  # Resize video frames to 64x64
    ])
    save_folder = 'F:\spo2_data_collected'
    num_frames = 150  # Define how many frames you want to capture and process
    # End Parameters ##################################################################################################
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Initialize FrameProcessor
    processor = FrameProcessor(transform)

    if capture:
        # Initialize capture from a video file or camera (e.g., 0 for the default camera)
        capture = cv2.VideoCapture(0)

        #frames_tensor = processor.get_data_from_capture(capture, num_frames)
        for frames_tensor in processor.get_data_from_capture(capture, save_adr=save_folder):
            print(frames_tensor.shape)

    else:
        spo2_csvs = [
            r"C:\Users\Crispy\AppData\Local\VirtualStore\Program Files (x86)\SpO2 Assistant V3.1.0.4\Data\_user_1_0_20240604021547-20240604021555joint.csv",
            r"C:\Users\Crispy\AppData\Local\VirtualStore\Program Files (x86)\SpO2 Assistant V3.1.0.4\Data\cristi_3am_user_1_1_20240604025918_298.csv",
            r"C:\Users\Crispy\AppData\Local\VirtualStore\Program Files (x86)\SpO2 Assistant V3.1.0.4\Data\_user_1_0_20240604152520-20240604152528joint.csv"
                        ]
        data_folder = r'F:\spo2_data_collected'
        processor.get_spo2_df(data_folder=data_folder, spo2_csvs=spo2_csvs, save_adr='collected_df.xlsx')



