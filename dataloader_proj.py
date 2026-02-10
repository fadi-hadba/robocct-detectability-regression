
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from natsort import natsorted
from PythonTools import raw2py
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import io

from PythonTools import py2raw
import matplotlib.pyplot as plt
import cv2

from torch.utils.data import DataLoader


class Detectability_Projections_Dataset(Dataset):
    def __init__(self, inputFolder, t_voxel_dict, input_nc, mode="train", **kwargs):

        # save self.images as a tensor of tensors. do the same with self.targets
    
        self.error_dict = t_voxel_dict

        # error_list = list(ERROR_DICT.values())
        self.image_size_resize = 375
        self.input_nc = input_nc
        self.inputFolder = inputFolder
        self.transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 65535)])
        self.images = []
        self.dir = []
        self.targets = []
        self.counter = 0

        self.process_subdirectories(self.inputFolder)

        for d in self.dir:
            # error_values = error_list[idx]
            object_name = os.path.basename(d)
            print(object_name)
            voxel_of_interest = [np.array(self.error_dict[object_name]) - np.round(375/2)] # meine Daten sind 375x375

            self.d = d
            self.paths_images = get_images_sorted(d)
            self.headers = []
            self.images.extend(self.loadProjections(self.paths_images, voxel_of_interest))

            if mode == "train":
                self.targets.extend(self.load_detectabilites("\\d_combined.npy"))

            else:

                self.targets.extend(np.zeros(len(self.images)))

        # # Konvertiere self.targets in eine Liste von Tensoren mit der Größe torch.Size([1])
        # self.targets = [torch.tensor([target]) for target in self.targets]

        # # Konvertiere die Liste von Tensoren in einen einzigen Tensor
        # self.targets = torch.cat(self.targets).unsqueeze(1)

        print("Length of targets:", len(self.targets))
        # plt.hist(self.targets.detach().numpy().flatten(), bins=50)
        # plt.title('Histogram of Tensor Values')
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # plt.show()

        #self.images = torch.stack(self.images)
        #indices = torch.nonzero(self.images.view(self.images.shape[0], -1).sum(dim=1) == 0).squeeze()
        #indices = list(indices.numpy() if torch.is_tensor(indices) else indices)

        #self.targets = [val for idx, val in enumerate(self.targets) if idx not in indices]
        #self.images = torch.stack([tensor for idx, tensor in enumerate(self.images) if idx not in indices])

        print("initialized dataset")

    def set_transformations(self, transform):
        self.transformation = transform

    def process_subdirectories(self, folder):
        for root, dirs, files in os.walk(folder):
            if "d_combined.npy" in files:
                self.dir.append(root)

    def loadProjections(self, projs_path, voxel_of_interest):
        img = list()
        for i, proj in enumerate(projs_path):
            header, image = raw2py.raw2py(proj, switch_order=True)

            #print(header)

            #projection_matrix = header_to_projection_matrix(header)

            #roi = extract_roi_in_projection(image, projection_matrix, voxel_of_interest)

            image = Image.fromarray(image)
            # Convert image to float before normalization
            image = image.convert('F')
            #necessary to use the transformation to tensor method!
            #image = image.point(lambda i:i*(1./256))
            img.append(self.transformation(image).float())


        return img


    def load_detectabilites(self, file_name):
        detectabilities = []
        path = self.d + file_name

        detec = np.load(path)
        detec = np.nan_to_num(detec)
        maximum = 1400186995.2000003
        for det in detec:
            normalized_det = np.log10(det / maximum + 0.1) + 4
            tensor_det = torch.tensor(normalized_det).float()  # Umwandlung in Tensor
            detectabilities.append(tensor_det)

        #Plot Histogram
        # plt.hist([tensor.item() for tensor in detectabilities], bins=50, color='blue', edgecolor='black')
        # plt.title('Histogram der tatsächlichen Werte')
        # plt.xlabelmb
        # plt.ylabel('Häufigkeit')
        # plt.show()


        # Gegebene Daten
        labels = ['ResNet50', 'ResNet152', 'EfficientNetB1', 'EfficientNetB7']
        values = [0.24, 8180, 12.82, 53288]

        # Erstelle Histogramm
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color='darkblue')

        # Setze logarithmische Skala für Y-Achse
        plt.yscale('log')

        # Beschriftungen und Titel
        plt.xlabel('Modelle mit unterschiedlichen Tiefen')
        plt.ylabel('Fehlerwerte')
        plt.title('Histogramm der Fehlerwerte von Testdatensatz')

        # Füge Werte über den Balken hinzu
        for i in range(len(labels)):
            plt.text(i, values[i], str(values[i]), ha='center', va='bottom')

        # Zeige das Histogramm
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return detectabilities

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self.images[index]


        target = self.targets[index]

        return image, target

    def __len__(self):
        """ Returns the size of the dataset. """

        return len(self.images)

    def name(self):
        return 'Detectability_Projections_Dataset'


def get_images_sorted(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in fnames:
            if "raw" in fname:
                path = os.path.join(root, fname)
                images.append(path)
    return natsorted(images)


def adapt_grayscale_channel_number(img, nc):
    width, height = img.size
    out = np.zeros((nc, height, width), dtype=np.float32)
    for i in range(0, nc):
        out[i, :, :] = np.array(img, dtype=np.float32)
    return torch.tensor(out)


def header_to_projection_matrix(header, voxel_size=[0.45, 0.45], swap_detector_axis=False, **kwargs):
    # Source: Auto-calibration of cone beam geometries from arbitrary rotating markers
    # using a vector geometry formulation of projection matrices by Graetz, Jonas

    detector_shape = np.array([header.number_vertical_pixels, header.number_horizontal_pixels])

    # Shift into left upper corner of the detector
    detector_left_corner_trans = np.eye(3)
    detector_left_corner_trans[0, 2] = + (float(header.number_vertical_pixels) - 1.) / 2.
    detector_left_corner_trans[1, 2] = + (float(header.number_horizontal_pixels) - 1.) / 2.
    detector_left_corner_trans[0, 0] *= 1
    detector_left_corner_trans[1, 1] *= -1
    detector_left_corner_trans[2, 2] = 1.

    traj_type = 'circ' if np.array_equal(np.array(header.agv_source_position), np.array([0, 0, 0])) else 'free'

    # Initial stuff for circular trajectory:
    if traj_type == 'circ':
        init_source_position = [0, 0, header.focus_object_distance_in_mm]
        init_detector_position = [0, 0, header.focus_object_distance_in_mm - header.focus_detector_distance_in_mm]
        init_detector_line_direction = [0, 1, 0]
        init_detector_column_direction = [1, 0, 0]
        angular_range = header.scan_range_in_rad
        if angular_range == 0:
            angular_range = 2 * np.pi

        current_angle = 0
        angular_increment = angular_range / 1  # Only one header, so only one projection matrix is needed
    else:
        det_h = np.array(header.agv_detector_line_direction)
        det_v = -1 * np.array(header.agv_detector_col_direction)
        source_center_in_voxel = (np.array(header.agv_source_position) / 1000) / voxel_size[0]  # in mm
        detector_center_in_voxel = (np.array(header.agv_detector_center_position) / 1000) / voxel_size[0]  # in mm

    if traj_type == 'free':
        det_h = np.array(header.agv_detector_line_direction)
        det_v = -1 * np.array(header.agv_detector_col_direction)
        source_center_in_voxel = (np.array(header.agv_source_position) / 1000) / voxel_size[0]  # in mm
        detector_center_in_voxel = (np.array(header.agv_detector_center_position) / 1000) / voxel_size[0]  # in mm
    else:
        # rotation about x-axis => Column direction of the detector
        R_x_axis = np.eye(3, 3)
        R_x_axis = np.array([1, 0, 0,
                             0, np.cos(-current_angle), -np.sin(-current_angle),
                             0, np.sin(-current_angle), np.cos(-current_angle)]).reshape((3, 3))
        source_center_in_voxel = np.dot(R_x_axis, init_source_position) / voxel_size[0]
        detector_center_in_voxel = np.dot(R_x_axis, init_detector_position) / voxel_size[0]
        det_h = np.dot(R_x_axis, init_detector_line_direction)
        det_v = np.dot(R_x_axis, init_detector_column_direction)
        current_angle += angular_increment

    # [H|V|d-s]
    h_v_sdd = np.column_stack((det_h, det_v, (detector_center_in_voxel - source_center_in_voxel)))
    h_v_sdd_invers = np.linalg.inv(h_v_sdd)

    # [H|V|d-s]^-1 * -s
    back_part = h_v_sdd_invers @ (-source_center_in_voxel)
    proj_matrix = np.column_stack((h_v_sdd_invers, back_part))
    projection_matrix = detector_left_corner_trans @ proj_matrix

    # Post-processing to get the same oriented output volume like ezrt command-line reco:
    # Flip Z-Axis: Z = -Z
    # projection_matrix[0:3, 2] = projection_matrix[0:3, 2] * -1.0

    # Change orientation of current matrix from XYZ to YXZ: exchange the first two columns
    # projection_matrix[0:3, 0:2] = np.flip(projection_matrix[0:3, 0:2], axis=1)

    return projection_matrix



def extract_roi_in_projection(projection, projection_matrix, voxels_of_interest, size_roi=[50, 50]):
    count = 0
    detector_shape = projection.shape

    for voxel in voxels_of_interest:
        pixel_on_detector_h = projection_matrix @ np.append(voxel, 1)
        pixel_on_detector = np.array((int(pixel_on_detector_h[0] / pixel_on_detector_h[2]),
                                      int(pixel_on_detector_h[1] / pixel_on_detector_h[2])))

        if np.any(pixel_on_detector < 0) or pixel_on_detector[0] >= detector_shape[0] or pixel_on_detector[1] >= \
                detector_shape[1]:
            print("Invalid pixel position")
            count += 1
            print(count)
            continue

        x_start = max(pixel_on_detector[0] - size_roi[0] // 2, 0)
        x_end = min(pixel_on_detector[0] + size_roi[0] // 2 , detector_shape[0])
        y_start = max(pixel_on_detector[1] - size_roi[1] // 2, 0)
        y_end = min(pixel_on_detector[1] + size_roi[1] // 2 , detector_shape[1])

        roi = projection[y_start:y_end, x_start:x_end]

         # Check if padding is needed
        pad_x = size_roi[0] - (x_end - x_start)
        pad_y = size_roi[1] - (y_end - y_start)

        # Add padding if necessary
        if pad_x > 0 or pad_y > 0:
            roi = np.pad(roi, ((max(0, pad_y // 2), max(0, pad_y - pad_y // 2)), 
                               (max(0, pad_x // 2), max(0, pad_x - pad_x // 2))),
                         mode='constant', constant_values=0)

    return roi
