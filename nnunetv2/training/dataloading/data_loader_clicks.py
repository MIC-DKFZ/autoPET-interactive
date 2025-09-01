import os
import warnings
from typing import Union, Tuple, List

import numpy as np
import torch
from skimage.morphology import ball
from threadpoolctl import threadpool_limits
from scipy.ndimage import gaussian_filter

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.utils import generated_sparse_to_dense_point_gauss, simulate_clicks, \
    sparse_to_dense_point_gauss, generated_sparse_to_dense_point_nnInteractive, simulate_clicks_advanced
from nnunetv2.utilities.label_handling.label_handling import LabelManager

class nnUNetDataLoaderClicks(nnUNetDataLoader):    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties, click_json = self._data.load_case_with_clicks(i)
            shape = data.shape[1:]

            # PROMPT HANDLING
            # Sample a random number of clicks from the click json
            num_clicks = np.random.randint(0, len(click_json['points']) + 1)
            clicks = np.random.choice(click_json['points'], size=num_clicks, replace=False)

            # Initialize the click volumes
            # pos_clicks, neg_clicks = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
            # if num_clicks > 0:
            #     clicks = np.random.choice(click_json['points'], size=num_clicks, replace=False)
            #     for clck in clicks:
            #         coord = clck['point']
            #         label = clck['name']
            #         coord = self.preprocess_point(coord, properties, shape)
            #         if label == 'tumor':
            #             # put point at the coordinate (not place_point)
            #             pos_clicks[*coord] = 1.0
            #         elif label == 'background':
            #             neg_clicks[*coord] = 1.0 # self.place_point(coord, neg_clicks, n_clck + 1)
            #         else:
            #             raise ValueError(f"Unknown label {label} in click json")
            #     pos_clicks = gaussian_filter(pos_clicks, sigma=3)
            #     neg_clicks = gaussian_filter(neg_clicks, sigma=3)

            pos_clicks, neg_clicks = sparse_to_dense_point_gauss(clicks, shape, properties, sigma=3)
                    
            # import napari
            # viewer = napari.Viewer()
            # viewer.add_image(data[0], name='CT')
            # viewer.add_image(data[1], name='PET')
            # viewer.add_labels(seg[0], name='segmentation')
            # viewer.add_labels(seg_prev[0], name='segmentation_prev')
            # viewer.add_labels(pos_clicks.astype(np.uint8), name='positive clicks')
            # viewer.add_labels(neg_clicks.astype(np.uint8), name='negative clicks')
            # napari.run()
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

            # combine clicks
            pos_clicks_cropped = crop_and_pad_nd(pos_clicks[None], bbox, 0)
            neg_clicks_cropped = crop_and_pad_nd(neg_clicks[None], bbox, 0)
            clicks_cropped = np.vstack((pos_clicks_cropped, neg_clicks_cropped))
            clicks_all[j] = clicks_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
            clicks_all = clicks_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    clicks_all = torch.from_numpy(clicks_all).float()
                    images = []
                    segs = []
                    clicks = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b], 'regression_target': clicks_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                        clicks.append(tmp['regression_target'])
                    data_all = torch.stack(images)
                    clicks_all = torch.stack(clicks)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(crop_and_pad_nd(data, bbox, 0)[0], name='CT original')
        # viewer.add_image(crop_and_pad_nd(data, bbox, 0)[1], name='PET original')
        # viewer.add_image(pos_clicks_cropped, name='positive clicks')
        # viewer.add_image(neg_clicks_cropped, name='negative clicks')
        # viewer.add_image(data_all[1][0].numpy(), name='CT')
        # viewer.add_image(data_all[1][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][1,0].numpy(), name='segmentation')
        # viewer.add_image(clicks_all[1][0].numpy(), name='positive clicks da')
        # viewer.add_image(clicks_all[1][1].numpy(), name='negative clicks da')
        # napari.run()
                    
        # Combine clicks and image
        data_all = torch.cat((data_all, clicks_all), dim=1)
        
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

class nnUNetDataLoaderClicksGenerated(nnUNetDataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None,
                 point_width: float = 1.5):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides,
                         probabilistic_oversampling, transforms)
        self.point_width = point_width    
        
    def generate_train_batch_full_img(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]

            # PROMPT HANDLING
            # Sample a random number of clicks from the click json
            num_clicks = np.random.randint(0, 10)
            
            clicks = simulate_clicks(seg[0], data[1], fg=True, bg=True, center_offset=3, edge_offset=3, click_budget=num_clicks, use_gpu=False)

            pos_clicks, neg_clicks = generated_sparse_to_dense_point_gauss(clicks, shape, sigma=self.point_width)
                    
            # import napari
            # viewer = napari.Viewer()
            # viewer.add_image(data[0], name='CT')
            # viewer.add_image(data[1], name='PET')
            # viewer.add_labels(seg[0], name='segmentation')
            # viewer.add_labels(seg_prev[0], name='segmentation_prev')
            # viewer.add_labels(pos_clicks.astype(np.uint8), name='positive clicks')
            # viewer.add_labels(neg_clicks.astype(np.uint8), name='negative clicks')
            # napari.run()
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

            # combine clicks
            pos_clicks_cropped = crop_and_pad_nd(pos_clicks[None], bbox, 0)
            neg_clicks_cropped = crop_and_pad_nd(neg_clicks[None], bbox, 0)
            clicks_cropped = np.vstack((pos_clicks_cropped, neg_clicks_cropped))
            clicks_all[j] = clicks_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
            clicks_all = clicks_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    clicks_all = torch.from_numpy(clicks_all).float()
                    images = []
                    segs = []
                    clicks = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b], 'regression_target': clicks_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                        clicks.append(tmp['regression_target'])
                    data_all = torch.stack(images)
                    clicks_all = torch.stack(clicks)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images, clicks

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0], name='CT original')
        # viewer.add_image(data[1], name='PET original')
        # viewer.add_image(pos_clicks, name='positive clicks')
        # viewer.add_image(neg_clicks, name='negative clicks')
        # viewer.add_image(data_all[1][0].numpy(), name='CT')
        # viewer.add_image(data_all[1][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][1,0].numpy(), name='segmentation')
        # viewer.add_image(clicks_all[1][0].numpy(), name='positive clicks da')
        # viewer.add_image(clicks_all[1][1].numpy(), name='negative clicks da')
        # napari.run()
                    
        # Combine clicks and image
        data_all = torch.cat((data_all, clicks_all), dim=1)
        
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})

                        num_pos_clicks, num_neg_clicks = np.random.randint(0, 6), np.random.randint(0, 6)
                        pet = tmp['image'][1].numpy()
                        clicks = simulate_clicks(tmp['segmentation'][0][0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                        pos_clicks, neg_clicks = generated_sparse_to_dense_point_gauss(clicks, pet.shape, sigma=self.point_width)
                        clicks_all = np.concat((pos_clicks[None], neg_clicks[None]), axis=0)
                        #clicks_arr.append(torch.from_numpy(clicks_all).float())

                        images.append(torch.cat((tmp['image'], torch.from_numpy(clicks_all).float()), dim=0))
                        segs.append(tmp['segmentation'])

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images # , clicks, clicks_all, pos_clicks, neg_clicks, tmp

                    # click_array = torch.zeros_like(data_all)
                    # for b in range(self.batch_size):
                    #     pet = data_all[b][1].numpy()
                    #     num_pos_clicks, num_neg_clicks = np.random.randint(0, 6), np.random.randint(0, 6)
                    #     clicks = simulate_clicks(seg_all[0][b,0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                    #     pos_clicks, neg_clicks = generated_sparse_to_dense_point_gauss(clicks, pet.shape, sigma=1.5)
                    #     clicks_all = np.concat((pos_clicks[None], neg_clicks[None]), axis=0)
                    #     click_array[b] = torch.from_numpy(clicks_all).float()
        
                    # data_all = torch.cat((data_all, click_array), dim=1)
                    # del clicks_all, pos_clicks, neg_clicks, clicks, click_array, pet

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0], name='CT original')
        # viewer.add_image(data[1], name='PET original')
        # viewer.add_image(data_all[1][0].numpy(), name='CT')
        # viewer.add_image(data_all[1][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][1,0].numpy(), name='segmentation')
        # viewer.add_image(data_all[1][2].numpy(), name='positive clicks da')
        # viewer.add_image(data_all[1][3].numpy(), name='negative clicks da')
        # napari.run()
                
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
    

class nnUNetDataLoaderClicksGeneratedEDT(nnUNetDataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None,
                 point_width: float = 1.5):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides,
                         probabilistic_oversampling, transforms)
        self.point_width = point_width
    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})

                        num_pos_clicks, num_neg_clicks = np.random.randint(0, 6), np.random.randint(0, 6)
                        pet = tmp['image'][1].numpy()
                        clicks = simulate_clicks(tmp['segmentation'][0][0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                        pos_clicks, neg_clicks = generated_sparse_to_dense_point_nnInteractive(clicks, pet.shape, sigma=self.point_width)
                        clicks_all = torch.cat((pos_clicks[None], neg_clicks[None]), axis=0).float()

                        images.append(torch.cat((tmp['image'], clicks_all), dim=0))
                        segs.append(tmp['segmentation'])

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images # , clicks, clicks_all, pos_clicks, neg_clicks, tmp

                    # click_array = torch.zeros_like(data_all)
                    # for b in range(self.batch_size):
                    #     pet = data_all[b][1].numpy()
                    #     num_pos_clicks, num_neg_clicks = np.random.randint(0, 6), np.random.randint(0, 6)
                    #     clicks = simulate_clicks(seg_all[0][b,0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                    #     pos_clicks, neg_clicks = generated_sparse_to_dense_point_gauss(clicks, pet.shape, sigma=1.5)
                    #     clicks_all = np.concat((pos_clicks[None], neg_clicks[None]), axis=0)
                    #     click_array[b] = torch.from_numpy(clicks_all).float()
        
                    # data_all = torch.cat((data_all, click_array), dim=1)
                    # del clicks_all, pos_clicks, neg_clicks, clicks, click_array, pet

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0], name='CT original')
        # viewer.add_image(data[1], name='PET original')
        # viewer.add_image(data_all[1][0].numpy(), name='CT')
        # viewer.add_image(data_all[1][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][1,0].numpy(), name='segmentation')
        # viewer.add_image(data_all[1][2].numpy(), name='positive clicks da')
        # viewer.add_image(data_all[1][3].numpy(), name='negative clicks da')
        # napari.run()
                
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
    

class nnUNetDataLoaderClicksGenerated10ptsEDT(nnUNetDataLoaderClicksGeneratedEDT):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        point_sampling_probs = np.log(np.linspace(2,12,11))[::-1]
        point_sampling_probs /= point_sampling_probs.sum()  # Normalize to sum to 1

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})

                        num_pos_clicks, num_neg_clicks = np.random.choice(np.arange(11), p=point_sampling_probs), np.random.choice(np.arange(11), p=point_sampling_probs)
                        pet = tmp['image'][1].numpy()
                        clicks = simulate_clicks(tmp['segmentation'][0][0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                        pos_clicks, neg_clicks = generated_sparse_to_dense_point_nnInteractive(clicks, pet.shape, sigma=self.point_width)
                        clicks_all = torch.cat((pos_clicks[None], neg_clicks[None]), axis=0).float()

                        images.append(torch.cat((tmp['image'], clicks_all), dim=0))
                        segs.append(tmp['segmentation'])

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images # , clicks, clicks_all, pos_clicks, neg_clicks, tmp

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0], name='CT original')
        # viewer.add_image(data[1], name='PET original')
        # viewer.add_image(data_all[1][0].numpy(), name='CT')
        # viewer.add_image(data_all[1][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][1,0].numpy(), name='segmentation')
        # viewer.add_image(data_all[1][2].numpy(), name='positive clicks da', colormap='green')
        # viewer.add_image(data_all[1][3].numpy(), name='negative clicks da', colormap='red')
        # napari.run()
                
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
    

class nnUNetDataLoaderClicksGenerated10ptsRatio80_20EDT(nnUNetDataLoaderClicksGeneratedEDT):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        point_sampling_probs = np.log(np.linspace(2,12,11))[::-1]
        point_sampling_probs /= point_sampling_probs.sum()  # Normalize to sum to 1

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})

                        num_pos_clicks, num_neg_clicks = np.random.choice(np.arange(11), p=point_sampling_probs), np.random.choice(np.arange(11), p=point_sampling_probs)
                        pet = tmp['image'][1].numpy()
                        if np.random.rand() < 0.8:  # 80% chance to use the normal click simulation
                            clicks = simulate_clicks(tmp['segmentation'][0][0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                        else:  # 20% chance to use the advanced click simulation
                            clicks = simulate_clicks_advanced(tmp['segmentation'][0][0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                        pos_clicks, neg_clicks = generated_sparse_to_dense_point_nnInteractive(clicks, pet.shape, sigma=self.point_width)
                        clicks_all = torch.cat((pos_clicks[None], neg_clicks[None]), axis=0).float()

                        images.append(torch.cat((tmp['image'], clicks_all), dim=0))
                        segs.append(tmp['segmentation'])

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images # , clicks, clicks_all, pos_clicks, neg_clicks, tmp

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0], name='CT original')
        # viewer.add_image(data[1], name='PET original')
        # viewer.add_image(data_all[0][0].numpy(), name='CT')
        # viewer.add_image(data_all[0][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][0,0].numpy(), name='segmentation')
        # viewer.add_image(data_all[0][2].numpy(), name='positive clicks da', colormap='green')
        # viewer.add_image(data_all[0][3].numpy(), name='negative clicks da', colormap='red')
        # napari.run()
                
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}