'''
process raw CT nii.gz and mask nii.gz into resampled, normalized, and cropped images (nii.gz)
'''
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import json
import SimpleITK as sitk
from skimage.morphology import binary_closing,binary_dilation
from skimage.measure import label,regionprops
import numpy as np
C3D = '/media/wukai/Models/c3d-1.4.2-Linux-gcc64/bin/c3d'

def resample_image_and_mask(img_path, out_dir):
    img_name = os.path.basename(img_path).replace('_0000.nii.gz', '')
    mask_path = os.path.join(os.path.dirname(img_path).replace('imagesTr', 'labelsTr'), img_name + '.nii.gz')
    out_img_dir = os.path.join(out_dir, img_name + '_resampled_0000.nii.gz')
    out_mask_dir = os.path.join(out_dir, img_name + '_resampled.nii.gz')
    os.system(f'{C3D} {img_path} -resample-mm 1x1x1mm -o {out_img_dir}')
    os.system(f'{C3D} {mask_path} -int 0 -resample-mm 1x1x1mm -o {out_mask_dir}')

def crop_image(img_path, crop_size, out_dir):
    img_name = os.path.basename(img_path).replace('_resampled_0000.nii.gz', '')
    mask_path = os.path.join(out_dir, img_name+'_signal_kidney_with_tumor.nii.gz')
    assert os.path.exists(mask_path), f'{mask_path} not exists'
    bbox_path = os.path.join(out_dir, img_name + '_crop_box.json')
    assert os.path.exists(bbox_path), f'{bbox_path} not exists'
    out_img_dir = os.path.join(out_dir, img_name + '_cropped_0000.nii.gz')
    out_mask_dir = os.path.join(out_dir, img_name + '_cropped.nii.gz')
    # load image and mask
    image_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)
    image = sitk.GetArrayFromImage(image_sitk)
    mask = sitk.GetArrayFromImage(mask_sitk)
    with open(bbox_path, 'r') as f:
        bbox = json.load(f)['bbox']
    # cal crop bbox
    center_point = [(bbox[3]+bbox[0])/2,(bbox[4]+bbox[1])/2,(bbox[5]+bbox[2])/2]
    crop_boxes = [coord - crop_size[i]/2 for i,coord in enumerate(center_point)] + [coord + crop_size[i]/2 for i,coord in enumerate(center_point)]

    # # 将image向外扩边，以防止剪裁时超边缘
    new_image = image_padding(image, crop_size=crop_size)
    new_mask = image_padding(mask, crop_size=crop_size)
    # # 由于原始image的H,Y,X轴向两侧均pading,crop_boxes坐标需要重新计算
    crop_boxes = boxes_padding(crop_boxes, crop_size=crop_size)
    # # 按crop坐标剪裁image
    image_cropped = crop_image_by_boxes(new_image, crop_boxes)
    mask_cropped = crop_image_by_boxes(new_mask, crop_boxes)
    # # remove the background 
    image_cropped[mask_cropped == 0] = np.min(image_cropped)
    # # 保存cropped image和mask
    image_cropped_sitk = sitk.GetImageFromArray(image_cropped)
    mask_cropped_sitk = sitk.GetImageFromArray(mask_cropped)
    copy_sitk_info(image_sitk, image_cropped_sitk)
    copy_sitk_info(mask_sitk, mask_cropped_sitk)
    sitk.WriteImage(image_cropped_sitk, out_img_dir)
    sitk.WriteImage(mask_cropped_sitk, out_mask_dir)


def crop_tumor_image(img_path, crop_size, out_dir):
    img_name = os.path.basename(img_path).replace('_resampled_0000.nii.gz', '')
    mask_path = os.path.join(out_dir, img_name+'_tumor.nii.gz')
    assert os.path.exists(mask_path), f'{mask_path} not exists'
    bbox_path = os.path.join(out_dir, img_name + '_tumor_box.json')
    assert os.path.exists(bbox_path), f'{bbox_path} not exists'
    out_img_dir = os.path.join(out_dir, img_name + '_tumor_cropped_0000.nii.gz')
    out_mask_dir = os.path.join(out_dir, img_name + '_tumor_cropped.nii.gz')
    # load image and mask
    image_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)
    image = sitk.GetArrayFromImage(image_sitk)
    mask = sitk.GetArrayFromImage(mask_sitk)
    with open(bbox_path, 'r') as f:
        bbox = json.load(f)['bbox']
    # cal crop bbox
    center_point = [(bbox[3]+bbox[0])/2,(bbox[4]+bbox[1])/2,(bbox[5]+bbox[2])/2]
    crop_boxes = [coord - crop_size[i]/2 for i,coord in enumerate(center_point)] + [coord + crop_size[i]/2 for i,coord in enumerate(center_point)]

    # # 将image向外扩边，以防止剪裁时超边缘
    new_image = image_padding(image, crop_size=crop_size)
    new_mask = image_padding(mask, crop_size=crop_size)
    # # 由于原始image的H,Y,X轴向两侧均pading,crop_boxes坐标需要重新计算
    crop_boxes = boxes_padding(crop_boxes, crop_size=crop_size)
    # # 按crop坐标剪裁image
    image_cropped = crop_image_by_boxes(new_image, crop_boxes)
    mask_cropped = crop_image_by_boxes(new_mask, crop_boxes)
    # # remove the background 
    image_cropped[mask_cropped == 0] = np.min(image_cropped)
    # # 保存cropped image和mask
    image_cropped_sitk = sitk.GetImageFromArray(image_cropped)
    mask_cropped_sitk = sitk.GetImageFromArray(mask_cropped)
    copy_sitk_info(image_sitk, image_cropped_sitk)
    copy_sitk_info(mask_sitk, mask_cropped_sitk)
    sitk.WriteImage(image_cropped_sitk, out_img_dir)
    sitk.WriteImage(mask_cropped_sitk, out_mask_dir)

def compute_kidney_with_tumor_coord(img_path, out_dir):
    img_name = os.path.basename(img_path).replace('_resampled_0000.nii.gz', '')
    mask_path = img_path.replace('_0000','')
    assert os.path.exists(mask_path), f'{mask_path} not exists'

    mask_sitk = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask_sitk)

    # 计算crop坐标: 肾+肿瘤
    new_mask, bbox, center_point = compute_crop_boxes_coord(mask)

    new_mask = new_mask.astype(np.uint8)
    new_mask_path = os.path.join(out_dir, img_name + '_signal_kidney_with_tumor.nii.gz')
    new_mask_sitk = sitk.GetImageFromArray(new_mask)
    new_mask_sitk.CopyInformation(mask_sitk)
    sitk.WriteImage(new_mask_sitk, new_mask_path)

    json_path = os.path.join(out_dir, img_name + '_crop_box.json')
    with open(json_path, 'w') as f:
        json.dump({'bbox': list(bbox), 'center_point': list(center_point)}, f)


def compute_crop_boxes_coord(mask):
    # 计算包含肿瘤的肾
    tumor_mask = np.zeros_like(mask)
    tumor_mask[mask == 2] = 1
    kidney_mask = np.zeros_like(mask)
    kidney_mask[mask == 1] = 1

    # tumor取最大的一个
    labels = label(tumor_mask)
    regions = regionprops(labels)
    tumor_index = np.argmax([region.area for region in regions])
    tumor_region = regions[tumor_index]
    # kidney取最大的前两个
    labels = label(kidney_mask)
    regions = regionprops(labels)
    kidney_index = np.argsort([region.area for region in regions])[::-1][:2]
    kidney_regions = [regions[i] for i in kidney_index]

    # 判断哪个肾为肿瘤所在肾
    for kidney_region in kidney_regions:
        intersection = compute_intersection_3D(tumor_region.bbox, kidney_region.bbox)
        if intersection > 0:
            kidney_tumor_loc_region = kidney_region
        else:
            kidney_tumor_loc_region = None

    # 合并肿瘤与病灶肾
    new_mask = np.zeros_like(mask)
    for coord in tumor_region.coords:
        new_mask[coord.tolist()[0],coord.tolist()[1],coord.tolist()[2]] = 2
    if kidney_tumor_loc_region is not None:
        for coord in kidney_tumor_loc_region.coords:
            new_mask[coord.tolist()[0],coord.tolist()[1],coord.tolist()[2]] = 1
    # 计算target区域中心点坐标
    
    labels =  label(binary_closing(new_mask>0))
    regions = regionprops(labels)
    assert len(regions) == 1, 'Wrong New mask with tumor and kidney'
    center_point = [int(i) for i in np.array(regions[0].centroid)] # 质心坐标
    bbox = regions[0].bbox # 边界框坐标
    # target_size = [bbox[3]-bbox[0],bbox[4]-bbox[1],bbox[5]-bbox[2]] # z, y x 
    # 计算需剪切box的坐标
    return new_mask, bbox, center_point

def compute_intersection_3D(box1, box2):
    # Calculate intersection areas
    z1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x1 = np.maximum(box1[2], box2[2])
    z2 = np.minimum(box1[3], box2[3])
    y2 = np.minimum(box1[4], box2[4])
    x2 = np.minimum(box1[5], box2[5])
    if (z2-z1) < 0:
        Z_distance = 0
    else:
        Z_distance = 1
    if (y2-y1) < 0:
        Y_distance = 0
    else:
        Y_distance = 1
    if (x2-x1) < 0:
        X_distance = 0
    else:
        X_distance = 1
    intersection = Z_distance*Y_distance*X_distance
    return intersection

def image_padding(array, crop_size = [120,120,120]):
    '''
    return a padded data array with cval
    在图像三个维度的最外边缘填充，值为cval
    '''
    pad_H, pad_Y, pad_X = [int(i/2) for i in crop_size]
    pad_width = ((pad_H, pad_H),(pad_Y, pad_Y),(pad_X, pad_X))
    new_array = np.pad(array, pad_width = pad_width, mode = 'constant', constant_values = np.min(array))
    return new_array

def boxes_padding(box, crop_size = [120,120,120]):
    pad_H, pad_Y, pad_X = [int(i/2) for i in crop_size]
    # boxes中每个轴的两个坐标均需要加上pad数值
    box[0], box[3] = box[0] + pad_H, box[3] + pad_H
    box[1], box[4] = box[1] + pad_Y, box[4] + pad_Y
    box[2], box[5] = box[2] + pad_X, box[5] + pad_X
    return box

def crop_image_by_boxes(image, crop_boxes):
    h1,y1,x1,h2,y2,x2 = [int(i) for i in crop_boxes]
    return image[h1:h2,y1:y2,x1:x2]

def stats_kidney_with_tumor_regions(file_list):
    target_size_list = []
    for json_path in tqdm(file_list, 'Analyzing crop box'):
        with open(json_path, 'r') as f:
            crop_box = json.load(f)
        bbox = crop_box['bbox']
        target_size = [bbox[3]-bbox[0],bbox[4]-bbox[1],bbox[5]-bbox[2]]
        target_size_list.append(target_size)
    target_size_array = np.array(target_size_list)
    print(f'Mean target size: {np.mean(target_size_array, axis=0)}, \
            STD: {np.std(target_size_array, axis=0)}, \
            Max target size: {np.max(target_size_array, axis=0)}, \
            95 Percentile target size: {np.percentile(target_size_array, 90, axis=0)}')

    
def copy_sitk_info(src_sitk, dst_sitk):
    dst_sitk.SetOrigin(src_sitk.GetOrigin())
    dst_sitk.SetDirection(src_sitk.GetDirection())
    dst_sitk.SetSpacing(src_sitk.GetSpacing())  

def stats_foreground_CT_values(img_list):
    ct_values = []
    for img_path in tqdm(img_list, 'Analyzing CT values'):
        mask_path = img_path.replace('_0000','')
        assert os.path.exists(mask_path), f'{mask_path} not exists'
        image_sitk = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image_sitk)
        mask_sitk = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask_sitk)
        ct_values.extend(image[mask>0].tolist())
    ct_values = np.array(ct_values)
    mean_intensity = np.mean(ct_values)
    std_intensity = np.std(ct_values)
    lower_bound = np.percentile(ct_values, 0.5)
    upper_bound = np.percentile(ct_values, 95.5)
    voxels_meta = {'mean': mean_intensity,'sd': std_intensity, 'percentile_00_5': lower_bound, 'percentile_99_5': upper_bound}
    print(f'Mean CT value: {mean_intensity}, STD: {std_intensity}, 0.5th percentile: {lower_bound}, 99.5th percentile: {upper_bound}')
    return voxels_meta

def normalize_image(img_path, voxels_meta, out_dir):
    img_name = os.path.basename(img_path).replace('_cropped_0000.nii.gz', '')
    out_img_dir = os.path.join(out_dir, img_name + '_normalized_0000.nii.gz')
    image_sitk = sitk.ReadImage(img_path)
    image = sitk.GetArrayFromImage(image_sitk)
    mean_intensity = voxels_meta['mean']
    std_intensity = voxels_meta['sd']
    lower_bound = voxels_meta['percentile_00_5']
    upper_bound = voxels_meta['percentile_99_5']
    
    image = np.clip(image, lower_bound, upper_bound) 
    image = (image - mean_intensity) / std_intensity
    image_sitk = sitk.GetImageFromArray(image)
    copy_sitk_info(image_sitk, image_sitk)
    sitk.WriteImage(image_sitk, out_img_dir)

def compute_tumor_coord(img_path, out_dir):
    img_name = os.path.basename(img_path).replace('_resampled_0000.nii.gz', '')
    mask_path = img_path.replace('_0000','')
    assert os.path.exists(mask_path), f'{mask_path} not exists'

    mask_sitk = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(mask_sitk)

    # 计算crop坐标: 肾+肿瘤
    new_mask, bbox, center_point = compute_tumor_boxes_coord(mask)

    new_mask = new_mask.astype(np.uint8)
    new_mask_path = os.path.join(out_dir, img_name + '_tumor.nii.gz')
    new_mask_sitk = sitk.GetImageFromArray(new_mask)
    new_mask_sitk.CopyInformation(mask_sitk)
    sitk.WriteImage(new_mask_sitk, new_mask_path)

    json_path = os.path.join(out_dir, img_name + '_tumor_box.json')
    with open(json_path, 'w') as f:
        json.dump({'bbox': list(bbox), 'center_point': list(center_point)}, f)


def compute_tumor_boxes_coord(mask):
    # 计算最大肿瘤的bbox
    tumor_mask = np.zeros_like(mask)
    tumor_mask[mask == 2] = 1
    labels = label(tumor_mask)
    regions = regionprops(labels)
    tumor_region = regions[np.argmax([i.area for i in regions])]
    tumor_bbox = tumor_region.bbox
    return tumor_mask, tumor_bbox, tumor_region.centroid


if __name__ == '__main__':
    # for phase 1 processing
    dataset_dir = './dataset_kidney_cancer_segmentation/nnUNet_raw/Dataset101_KidneyCancerPhase2'
    output_dir = './dataset_kidney_cancer_classification/Dataset101_KidneyCancerPhase2'
    os.makedirs(output_dir, exist_ok=True)
    original_img_dir = os.path.join(dataset_dir, 'imagesTr')
    original_label_dir = os.path.join(dataset_dir, 'labelsTr')
    resampled_dir = os.path.join(output_dir, 'resampled')
    normalized_dir = os.path.join(output_dir, 'normalized')
    cropped_dir = os.path.join(output_dir, 'cropped')
    tumor_cropped_dir = os.path.join(output_dir, 'cropped_tumor2')
    tumor_normalized_dir = os.path.join(output_dir, 'normalized_tumor2')
    os.makedirs(tumor_normalized_dir, exist_ok=True)
    os.makedirs(tumor_cropped_dir, exist_ok=True)
    os.makedirs(resampled_dir, exist_ok=True)
    os.makedirs(normalized_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    # 1. resample images
    img_list = glob.glob(os.path.join(original_img_dir, '*_0000.nii.gz'))
    with Pool(processes=16) as pool:
        for i in tqdm(pool.imap_unordered(partial(resample_image_and_mask, out_dir=resampled_dir), img_list), total=len(img_list)):
            pass

    # 2. compute kidney with tumor region
    img_list = glob.glob(os.path.join(resampled_dir, '*_resampled_0000.nii.gz'))
    img_list.sort()
    # for img_path in tqdm(img_list):
    #     compute_kidney_with_tumor_coord(img_path, cropped_dir)
    with Pool(processes=32) as pool:
        for i in tqdm(pool.imap_unordered(partial(compute_kidney_with_tumor_coord, out_dir=cropped_dir), img_list), total=len(img_list)):
            pass

    # 3. stats of crop box
    file_list = glob.glob(os.path.join(cropped_dir, '*_crop_box.json'))
    file_list.sort()
    stats_kidney_with_tumor_regions(file_list)
    print('I choose the ~95 percentile target size: [128, 96, 96]')
    # 4. crop images
    crop_size = [128,96,96]
    img_list = glob.glob(os.path.join(resampled_dir, '*_resampled_0000.nii.gz'))
    img_list.sort()
    # for img_path in tqdm(img_list):
    #     crop_image(img_path, crop_size, cropped_dir)
    with Pool(processes=32) as pool:
        for i in tqdm(pool.imap_unordered(partial(crop_image, crop_size = crop_size, out_dir=cropped_dir), img_list), total=len(img_list)):
            pass

    #5. normalize images based on mean and std of foreground
    ## 5.1 compute mean and std of foreground
    img_list = glob.glob(os.path.join(cropped_dir, '*_cropped_0000.nii.gz'))
    img_list.sort()
    voxels_meta = stats_foreground_CT_values(img_list)
    with open(os.path.join(normalized_dir, 'voxels_meta.json'), 'w') as f:
        json.dump(voxels_meta, f)
    ## 5.2 normalize images
    assert os.path.exists(os.path.join(normalized_dir, 'voxels_meta.json')), 'voxels_meta.json not exists'
    with open(os.path.join(normalized_dir, 'voxels_meta.json'), 'r') as f:
        voxels_meta = json.load(f)
    # for img_path in tqdm(img_list):
    #     normalize_image(img_path, voxels_meta, normalized_dir)
    with Pool(processes=32) as pool:
        for i in tqdm(pool.imap_unordered(partial(normalize_image, voxels_meta=voxels_meta, out_dir=normalized_dir), img_list), total=len(img_list)):
            pass

    ########################################################################3
    # 2. compute tumor region
    img_list = glob.glob(os.path.join(resampled_dir, '*_resampled_0000.nii.gz'))
    img_list.sort()
    # for img_path in tqdm(img_list):
    #     compute_tumor_coord(img_path, cropped_dir)
    with Pool(processes=32) as pool:
        for i in tqdm(pool.imap_unordered(partial(compute_tumor_coord, out_dir=tumor_cropped_dir), img_list), total=len(img_list)):
            pass
    # 3. stats of crop box
    file_list = glob.glob(os.path.join(tumor_cropped_dir, '*_tumor_box.json'))
    file_list.sort()
    stats_kidney_with_tumor_regions(file_list)
    print('I choose the ~95 percentile target size: [80, 80, 80]')

    crop_size = [80,80,80]
    img_list = glob.glob(os.path.join(resampled_dir, '*_resampled_0000.nii.gz'))
    img_list.sort()

    # for img_path in tqdm(img_list):
    #     crop_tumor_image(img_path, crop_size, tumor_cropped_dir)
    with Pool(processes=32) as pool:
        for i in tqdm(pool.imap_unordered(partial(crop_tumor_image, crop_size = crop_size, out_dir=tumor_cropped_dir), img_list), total=len(img_list)):
            pass

    ## 5. normalize images based on mean and std of foreground
    # 5.1 compute mean and std of foreground

    img_list = glob.glob(os.path.join(tumor_cropped_dir, '*_tumor_cropped_0000.nii.gz'))
    img_list.sort()
    voxels_meta = stats_foreground_CT_values(img_list)
    with open(os.path.join(tumor_normalized_dir, 'voxels_meta.json'), 'w') as f:
        json.dump(voxels_meta, f)
    ## 5.2 normalize images
    assert os.path.exists(os.path.join(tumor_normalized_dir, 'voxels_meta.json')), 'voxels_meta.json not exists'
    with open(os.path.join(tumor_normalized_dir, 'voxels_meta.json'), 'r') as f:
        voxels_meta = json.load(f)
    # for img_path in tqdm(img_list):
    #     normalize_image(img_path, voxels_meta, tumor_normalized_dir)
    with Pool(processes=32) as pool:
        for i in tqdm(pool.imap_unordered(partial(normalize_image, voxels_meta=voxels_meta, out_dir=tumor_normalized_dir), img_list), total=len(img_list)):
            pass