import numpy as np
import cv2
import random
from random import randint
import albumentations as A
import glob
import os
from pathlib import Path


def bg_change(image, list_of_textures):
    '''
    Function for augmenting image by replacing the white background with a texture
    '''
    texture = random.choice(list_of_textures)
    angle = random.choice([0, 1, 2, 3])
    for _ in range(angle):
        texture = cv2.rotate(texture, cv2.ROTATE_90_CLOCKWISE)
    texture = cv2.resize(texture, (image.shape[1], image.shape[0]))

    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 240, 255, cv2.THRESH_BINARY)
    
    image_new_bg = cv2.bitwise_and(texture, image, mask=alpha)
    return image_new_bg


def augmentation_watermarks(img, watermark_path='./augmentation_results/watermarks'):
    '''
    Function for augmenting image by adding a watermark
    '''
    watermarks_list = []
    for filepath in glob.glob(os.path.join(watermark_path, '*.jpeg')):
        watermarks_list.append(cv2.imread(filepath))

    watermark = random.choice(watermarks_list)
    watermark = cv2.resize(watermark, (img.shape[1], img.shape[0]))

    result = cv2.addWeighted(img, 0.6, watermark, 0.35, 0)

    return result


def augmentation_axis(img):
    '''
    Function for augmenting image by adding axes
    '''
    x = 0
    dash_x = img.shape[1] // randint(2, 6)
    y = 0
    dash_y = 40 + randint(-10, 20)
    while x + dash_x <= img.shape[1]:
        while y + dash_y <= img.shape[0]:
            line = cv2.line(img, (x, y), (x, y + dash_y), (125, 125, 125), thickness=2)
            circle = cv2.circle(img, (x + dash_x, y + dash_y + dash_y // 2), 2, (125, 125, 125), thickness=2)
            result = cv2.bitwise_and(line, img)
            result = cv2.bitwise_and(circle, img)
            y += 2*dash_y
        x += dash_x
        y = 0
    while y + dash_y <= img.shape[0]:
        line = cv2.line(img, (x, y), (x, y + dash_y), (125, 125, 125), thickness=2)
        result = cv2.bitwise_and(line, img)
        y += 2*dash_y
        
    x = 0
    y = 0
    dash_x = 40 + randint(-10, 20)
    dash_y = img.shape[0] // randint(2,6)
    while y + dash_y <= img.shape[0]:
        while x + dash_x <= img.shape[1]:
            line = cv2.line(img, (x, y), (x + dash_x, y), (125, 125, 125), thickness=2)
            circle = cv2.circle(img, (x + dash_x + dash_x // 2, y + dash_y), 2, (125, 125, 125), thickness=2)
            result = cv2.bitwise_and(line, img)
            result = cv2.bitwise_and(circle, img)
            x += 2*dash_x
        y += dash_y
        x = 0
    while x+dash_x <= img.shape[1]:
        line = cv2.line(img, (x, y), (x + dash_x, y), (125, 125, 125), thickness=2)
        result = cv2.bitwise_and(line, img)
        x += 2*dash_x   

    return result


def blur(image, image2, list_of_textures):
    '''
    Function for augmenting image by blurring
    '''
    if random.choice([1,0,0,0,0]) == 1:
        image = bg_change(image2, list_of_textures) 
    if randint(1, 100) <= 10:
        image = augmentation_axis(image2)         
    if randint(1, 100) <= 10:
        image = augmentation_watermarks(image2)   
    transform = A.Blur(blur_limit=15, always_apply=True)
    augmented_image = transform(image=image)['image']
    return augmented_image


def pixelize(image, image2, list_of_textures):
    '''
    Function for augmenting image by pixelization
    '''
    if random.choice([1, 0, 0, 0, 0]) == 1:
        image = bg_change(image2, list_of_textures)
    if randint(1, 100) <= 10:
        image = augmentation_axis(image2)         
    if randint(1, 100) <= 10:
        image = augmentation_watermarks(image2)          
    transform = A.Downscale(scale_min=0.25, scale_max=0.35, interpolation=cv2.INTER_CUBIC, always_apply=True, p=0.5)
    augmented_image = transform(image=image)['image']
    return augmented_image


def change_bc(image, image2, list_of_textures):
    '''
    Function for augmenting image by changing the contrast (scan effect)
    '''
    if random.choice([1,0,0,0,0]) == 1:
        image = bg_change(image2, list_of_textures)
    if randint(1, 100) <= 10:
        image = augmentation_axis(image2)         
    if randint(1, 100) <= 10:
        image = augmentation_watermarks(image2)          
    transform = A.RandomBrightnessContrast(brightness_limit=[0, 0.6], contrast_limit=0.3, brightness_by_max=False, always_apply=True, p=0.5)
    augmented_image = transform(image=image)['image']
    return augmented_image


def noize(image, image2, list_of_textures):
    '''
    Function for augmenting image by adding noize
    '''
    if random.choice([1, 0, 0, 0, 0]) == 1:
        image = bg_change(image2, list_of_textures)
    if randint(1, 100) <= 10:
        image = augmentation_axis(image)         
    if randint(1, 100) <= 10:
        image = augmentation_watermarks(image)          
    transform = A.RandomRain(
        slant_lower=-10, slant_upper=10, drop_length=20,
        drop_width=2, drop_color=(200, 200, 200), blur_value=5,
        brightness_coefficient=1, rain_type='drizzle', always_apply=True, p=0.5)
    augmented_image = transform(image=image)['image']
    return augmented_image


def invert_image(image):
    '''
    Function for augmenting image by inverting
    '''
    augmented_image = cv2.bitwise_not(image)
    return augmented_image


def augment_image(
        image_path='./augmentation_results/img_comb/5_0_0_0.png',
        mask_path='./augmentation_results/mask_comb/5_0_0_0.png',
        out_path_img='./augmentation_results/augmented_images',
        out_path_mask='./augmentation_results/augmented_masks'):
    '''
    Function for augmenting image in random ways
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)

    base_image_name = os.path.splitext(os.path.basename(image_path))[0]
    base_mask_name = os.path.splitext(os.path.basename(mask_path))[0]

    list_of_textures = []
    for filepath in glob.glob('./augmentation_results/textures/*.jpeg'):
        list_of_textures.append(cv2.imread(filepath))

    list_of_functions = [blur, pixelize, change_bc, noize, noize]
    list_of_functions = [blur, noize, change_bc]

    for i in range(4):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        image2 = image.copy()
        result = image.copy()
        if randint(1, 100) <= 1:
            result = invert_image(result)
            pass
        else:
            result = random.choice(list_of_functions)(result, image2, list_of_textures)
        image_name = os.path.join(out_path_img, f'{base_image_name}_{i}_.png')
        mask_name = os.path.join(out_path_mask, f'{base_mask_name}_{i}_.png')
        cv2.imwrite(image_name, result)
        cv2.imwrite(mask_name, mask)


def augment_image_rotation(
        image_path='./augmentation_results/img_comb/5_0_0_0.png',
        mask_path='./augmentation_results/mask_comb/5_0_0_0.png',
        out_path_img='./augmentation_results/augmented_images',
        out_path_mask='./augmentation_results/augmented_masks'):
    '''
    Function for augmenting image by rotation
    '''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)

    base_image_name = os.path.splitext(os.path.basename(image_path))[0]
    base_mask_name = os.path.splitext(os.path.basename(mask_path))[0]

    for i in range(4):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        # image2 = image.copy()
        result = image.copy()  
        image_name = os.path.join(out_path_img, f'{base_image_name}_{i}_.png')
        mask_name = os.path.join(out_path_mask, f'{base_mask_name}_{i}_.png')
        cv2.imwrite(image_name, result)
        cv2.imwrite(mask_name, mask)


def texture_to_img(
        style_dict,
        image_path='./augmentation_results/img_comb/5_0_0_0.png',
        mask_path='./augmentation_results/mask_comb/5_0_0_0.png',
        out_path='./augmentation_results/augmented_images/'):
    '''
    Function for applying textures to the image by mask
    in accordance with the specified style: empty, gray, line_color, lining
    '''
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    out_image_name = out_path

    list_of_textures = []
    for filepath in glob.glob('./textures/walls/*.jpg'):
        list_of_textures.append(cv2.imread(filepath))
    
    texture = random.choice(list_of_textures)
    if style_dict['wall_type'] == "empty":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif style_dict['wall_type'] == "grey":
        
        texture = np.full((image.shape[1], image.shape[0], 3), (125, 125, 125), dtype='uint8')
        texture = cv2.resize(texture, (image.shape[1], image.shape[0]))
        tmp = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 240, 255, cv2.THRESH_BINARY)
    
        image_new_bg = cv2.bitwise_and(texture, image, mask=alpha)
        image_new_bg = cv2.bitwise_not(image_new_bg, image, mask=alpha)

        image = image_new_bg
    elif style_dict['wall_type'] == "line_color":

        texture = np.full((image.shape[1], image.shape[0], 3), style_dict['lines_color'], dtype='uint8')
        texture = cv2.resize(texture, (image.shape[1], image.shape[0]))
        tmp = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 240, 255, cv2.THRESH_BINARY)
    
        image_new_bg = cv2.bitwise_and(texture, image, mask=alpha)
        image_new_bg = cv2.bitwise_not(image_new_bg, image, mask=alpha)

        image = image_new_bg
    elif style_dict['wall_type'] == "lining":

        texture = cv2.resize(texture, (texture.shape[1] // 6, texture.shape[0] // 6))
        delta_y = (image.shape[1] - texture.shape[1]) // 2
        delta_x = (image.shape[0] - texture.shape[0]) // 2
        
        texture = cv2.copyMakeBorder(texture, delta_x, delta_x, delta_y, delta_y, cv2.BORDER_WRAP)
        texture = cv2.resize(texture, (image.shape[1], image.shape[0]))

        tmp = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 240, 255, cv2.THRESH_BINARY)

        image_new_bg = cv2.bitwise_xor(texture, image, mask=alpha)
        image_new_bg = cv2.bitwise_not(image_new_bg, image, mask=alpha)

        image = image_new_bg

    cv2.imwrite(out_image_name, image)

if __name__ == '__main__':
    augment_image()
    # augment_image_rotation()