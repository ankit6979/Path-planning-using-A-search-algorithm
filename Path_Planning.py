#Black defines obstacles , white path and coloured objects
#Returns occupied grids and planned path
#For each object a matching object which is nearest to it is found using ssim
import cv2
import numpy as np
import time
from skimage.measure import compare_ssim as ssim
import astarsearch
import traversal
import matplotlib.pyplot as plt

def main(file_name):
    image = cv2.imread(file_name)
    #plt.imshow(image)
    #plt.show()
    occupied_grids = []
    planned_path = {}

    (winW, winH) = (60, 60)
    obstacles = []
    index = [1, 1]

    #Create a blank image initialized a matrix of 0's
    blank_image = np.zeros((60,60,3), np.uint8)

    #Create an array of 100 blank images as size of map is 600*600
    list_images = [[blank_image for i in range(10)] for i in range(10)]
    maze = [[0 for i in range(10)] for i in range(10)]

    #Traversal time
    for (x, y, window) in traversal.sliding_window(image, stepSize = 60, windowSize = (winW, winH)):
        #print(x , end = ' ')
        #print(y)
        if(window.shape[0] != winH or window.shape[1] != winW):
            continue
        #Print index image is our iterator
        clone = image.copy()
        #Format square for opencv
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 255, 0), 4)
        #Crop the image
        crop_img = image[y:y + winW,  x:x + winH]
        #Add it to the array of images
        list_images[index[0] - 1][index[1] - 1] = crop_img.copy()
        #cv2.imshow('win', list_images[index[0] - 1][index[1] - 1])
        #time.sleep(0.25)

        #Print the occupied grids check if its white or not
        average_color_per_row = np.average(crop_img, axis = 0)
        average_color = np.average(average_color_per_row , axis = 0)
        average_color = np.uint8(average_color)

        #Iterate through the color matrix
        if(any(i <= 240 for i in average_color)):
            maze[index[1] - 1][index[0] - 1] = 1
            #If not majority white
            occupied_grids.append(tuple(index))
        if(any(i <= 20 for i in average_color)):
            obstacles.append(tuple(index))

        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.25)

        index[1] = index[1] + 1
        if(index[1] > 10):
            index[0] = index[0] + 1
            index[1] = 1

        #Shortest path
        #Get the list of objects
    
    list_colored_grids = [n for n in occupied_grids if n not in obstacles]

    for startimage in list_colored_grids:
        key_startimage = startimage
        img1 = list_images[startimage[0] - 1][startimage[1] - 1]
        for grid in [n for n in list_colored_grids if n != startimage]:
                #Next image
            img = list_images[grid[0] - 1][grid[1] - 1]
                #Convert to grayscale
            image =cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            image2 =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #Comapre structural similarity
            s = ssim(image, image2)
                #If they are similar perform a star search
            if(s > 0.9):
                result = astarsearch.astar(maze, (startimage[0] - 1, startimage[1] - 1), (grid[0] - 1, grid[1] - 1))
                list2 = []
                for t in result:
                    x, y = t[0], t[1]
                    list2.append(tuple((x+1, y+1)))
                    #result = list(list2[1:-1])
                if not result:
                    planned_path[startimage] = list(["No path", [], 0])
                planned_path[startimage] = list([str(grid), result, len(result) + 1])
    for obj in list_colored_grids:
        if not(obj in planned_path):
                planned_path[obj] = list(["No match", [], 0])
    return occupied_grids, planned_path
    
if( __name__ == '__main__'):
    image_filename = "test_image3.jpg"
    occupied_grids, planned_path = main(image_filename)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
