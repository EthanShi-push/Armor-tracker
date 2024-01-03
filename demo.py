# author   ：Tian.Z.L, SHI.Y
# time   ：2023/2/17  18:11
# file   ：armortracker

import cv2
import numpy as np
import math
import traceback
from matplotlib import pyplot as plt

 
def img_show(name, src):
    cv2.imshow(name, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_light(src, color = "Blue"):
    if color not in ["Red", "Blue"]:
        raise ValueError("Invalid color! Color must be 'Red' or 'Blue'.")
    blue, g, red = cv2.split(src)  # split channels
    # to do 自动阈值
    if color == "Blue":
        ret2, binary = cv2.threshold(blue, 224, 255, 0) # Binarize the image (identify blue light bar)
    elif color == "Red":
        ret2, binary = cv2.threshold(red, 224, 255, 0) # Binarize the image (identify red light bar)

    # # add open opeartor
    kernel = np.ones((3, 3),np.uint8)
    binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
    Gaussian = cv2.GaussianBlur(binary, (5, 5), 0)  # Gauss filter(noise reduction)
    # edge = cv2.Canny(binary, 50, 150)  # edge detect
    cv2.imshow("binary",Gaussian)
    draw_img = Gaussian.copy()
    whole_h, whole_w = binary.shape[:2] # the height and width of the img
    # find the contours of the img in tree mode
    contours, hierarchy = cv2.findContours(image=draw_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True) #  sort the contour based on the area

    width_array = []
    height_array = []
    point_array = []
    for cont in contours[:5]:# select the 5 contours with the largest area
        # The upper left corner is the origin
        x, y, w, h = cv2.boundingRect(cont) # bounding rect features of those contours
        try:
            if h / w >= 2 and h / whole_h > 0.1 and h > w: # find spatial relation bw light bars(tricky)
                # if height / h > 0.05:
                width_array.append(w)
                height_array.append(h)
                point_array.append([x, y])
        except:
            continue

    # calculate all rectangle area
    areas = [width_array[i] * height_array[i] for i in range(len(width_array))]
    areas_with_index = [(area, index) for index, area in enumerate(areas)]
    # sort the elements by area
    areas_with_index.sort()
    min_diff = float('inf')
    point_near = [0, 0]
    # find the smallest difference of two adjacent elements
    for i in range(1, len(areas_with_index)):
        diff = abs(areas_with_index[i][0] - areas_with_index[i-1][0])
        if diff < min_diff:
            min_diff = diff
            point_near[0] = areas_with_index[i-1][1]
            point_near[1] = areas_with_index[i][1]

    # calculate 4 vertices of the armor plate
    rectangle1 = point_array[point_near[0]]
    rectangle2 = point_array[point_near[1]]
    # Distinguish between left and right
    (left_rectangle, left_index), (right_rectangle, right_index) = ((rectangle1, point_near[0]), (rectangle2, point_near[1])) if rectangle1[0] < rectangle2[0] else ((rectangle2, point_near[1]), (rectangle1, point_near[0]))

    left_top = [left_rectangle[0] + width_array[left_index] / 2, left_rectangle[1] ]
    left_bottom = [left_rectangle[0] + width_array[left_index] / 2, left_rectangle[1] + height_array[left_index]]
    right_top = [right_rectangle[0] + width_array[right_index] / 2, right_rectangle[1] ]
    right_bottom = [right_rectangle[0] + width_array[right_index] / 2, right_rectangle[1] + height_array[right_index]]
    # print(left_top, left_bottom, right_top, right_bottom)
    return [left_top, left_bottom, right_top, right_bottom]

def extractNumbers(src, armor):
    # # Light length in image
    # light_length = 12
    # # Image size after warp
    # warp_height = 28
    # small_armor_width = 32
    # large_armor_width = 54
    # # Number ROI size
    # roi_size = (20, 28)
    #
    # # Warp perspective transform
    # lights_vertices = np.float32([armor['left_light']['bottom'], armor['left_light']['top'], armor['right_light']['top'], armor['right_light']['bottom']])
    #
    # top_light_y = (warp_height - light_length) // 2 - 1
    # bottom_light_y = top_light_y + light_length
    # warp_width = small_armor_width if armor['type'] == 'SMALL' else large_armor_width
    # target_vertices = np.float32([(0, bottom_light_y), (0, top_light_y), (warp_width - 1, top_light_y), (warp_width - 1, bottom_light_y)])
    #
    # rotation_matrix = cv2.getPerspectiveTransform(lights_vertices, target_vertices)
    # number_image = cv2.warpPerspective(src, rotation_matrix, (warp_width, warp_height))
    # # Get ROI
    #number_image = number_image[(warp_width - roi_size[0]) // 2 : (warp_width + roi_size[0]) // 2, :]
    #
    # # Binarize
    # number_image = cv2.cvtColor(number_image, cv2.COLOR_RGB2GRAY)
    # _, number_image = cv2.threshold(number_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #
    # return number_image

    light_length = 12
    warp_height = 28
    small_armor_width = 32
    large_armor_width = 54
    roi_size = (20, 28)

    left_light_top = armor['left_light']['top']
    left_light_bottom = armor['left_light']['bottom']
    right_light_top = armor['right_light']['top']
    right_light_bottom = armor['right_light']['bottom']
    armor_type = armor['type']

    # Calculate target vertices for perspective transformation
    top_light_y = (warp_height - light_length) // 2 - 1
    bottom_light_y = top_light_y + light_length
    warp_width = small_armor_width if armor_type == 'SMALL' else large_armor_width
    target_vertices = np.float32([(0, bottom_light_y), (0, top_light_y), (warp_width - 1, top_light_y), (warp_width - 1, bottom_light_y)])

    # Define source vertices based on the input 'armor' information
    source_vertices = np.float32([left_light_bottom, left_light_top, right_light_top, right_light_bottom])

    # Calculate the perspective transformation matrix
    rotation_matrix = cv2.getPerspectiveTransform(source_vertices, target_vertices)

    # Apply perspective transformation to the source image
    transformed_image = cv2.warpPerspective(src, rotation_matrix, (warp_width, warp_height))

    # Crop the region of interest (ROI) from the transformed image
    left_roi_x = (warp_width - roi_size[0]) // 2
    right_roi_x = left_roi_x + roi_size[0]
    roi = transformed_image[:, left_roi_x:right_roi_x]

    # Convert the ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Binarize the ROI using Otsu's thresholding
    _, binarized_roi = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return binarized_roi

# pic_path = "./Pictures/4.png" # Picture file path
# # (137, 147, 3)
# img = cv2.imread(pic_path, cv2.IMREAD_COLOR) # BGR mode
# position_list = detect_light(img, color = "Blue")
# armor = {
#     "left_light":{
#         "top":position_list[0],
#         "bottom":position_list[1],
#     },
#     "right_light":{
#         "top":position_list[2],
#         "bottom":position_list[3],
#     },
#     "type": 'SMALL',
# }
# number_image = extractNumbers(img, armor)
# plt.imshow(number_image, "gray")
# plt.show()
# exit()

if __name__ == "__main__":
    video = cv2.VideoCapture(r'1234567.mp4')
    while True:
        ret, img = video.read() # get frames
        try:
            position_list = detect_light(img)
        except:
            continue
        armor = {
            "left_light":{
                "top":position_list[0],
                "bottom":position_list[1],
            },
            "right_light":{
                "top":position_list[2],
                "bottom":position_list[3],
            },
            "type": 'SMALL',
        }

        number_image = extractNumbers(img, armor)
        cv2.imshow('Number', number_image)
        # img_show("1", number_image)
        x = np.array(position_list, np.int32)
        box = x.reshape((-1, 1, 2)).astype(np.int32) # use box to draw lines for the found armor plate
        cv2.polylines(img, [box], True, (0, 255, 0), 2)
        cv2.imshow('name', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # release
    video.release()
    cv2.destroyAllWindows()
