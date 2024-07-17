import cv2
import os
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

# 1202057 yousef sharbi
# 1202699 salah abuawada 

# list to store scores (2d)
scores_matrix = []

# Compare between test data and training data
for file_test in os.listdir("test_data"):
    # cv2: to read image file, and then store it on test_image
    test_image = cv2.imread("./test_data/" + file_test)

    # list to store score for row(test image with all training image)
    scores = []

    for file_training in os.listdir("training_data"):
        # cv2: to read image file, and then store it on training_image
        training_image = cv2.imread("./training_data/" + file_training)

        # SIFT (Scale-Invariant Feature Transform) algorithm, method used for extracting distinctive features from images(keypoints, descriptors)
        sift = cv2.SIFT_create() # create object sift to detecting keypoints and computing descriptors(provide representation of image content).
        kp1, des1 = sift.detectAndCompute(test_image, None) # detects keypoints and computes descriptors for the test image
        kp2, des2 = sift.detectAndCompute(training_image, None) # detects keypoints and computes descriptors for the training image
        
        # Brute-Force Matcher(compare each descriptor from des1 with each descriptor from des2)
        bf = cv2.BFMatcher()

        # KnnMatch(k-nearest neighbors) for each descriptor des1 there is two nearest descriptors des2(k=2)
        matches = bf.knnMatch(des1, des2, k=2)

        # If the distance of the nearest neighbor (m) is less than (0.75 times the distance of the second nearest neighbor (n)) --> good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Calculate score using (Euclidean) distance metrics
        current_score = 0
        for match in good_matches:
            # Get the coordinates of matched keypoints
            src_pt = kp1[match.queryIdx].pt # get the keypoint from the kp1 list using the index match.queryIdx
            dst_pt = kp2[match.trainIdx].pt # get the keypoint from the kp2 list using the index match.trainIdx
            
            # Calculate Euclidean distance between keypoints
            distance = ((src_pt[0] - dst_pt[0]) ** 2 + (src_pt[1] - dst_pt[1]) ** 2) ** 0.5
            
            current_score += distance

        # add score for this image to list (add the scores on row)
        scores.append(current_score)

    # add the rows to the matrix
    scores_matrix.append(scores)


# Print the scores matrix
test_values = [f"Test {i+1}" for i in range(len(scores_matrix))]
training_values = [f"Training (Templ) {i+1}" for i in range(len(scores_matrix[0]))]

print("Scores Matrix:")
print(tabulate(scores_matrix, headers=training_values, showindex=test_values, tablefmt="grid"))

# #-------------------------------------------



list_threshould=[1117, 1500, 1992, 2000, 3901, 4500, 5748]

list_FNMR=[]
list_FMR=[]



for th in list_threshould:

    count_fnmr=0
    count_fmr=0

    for i in range(len(scores_matrix)):
        for j in range(len(scores_matrix[i])):
            if(i==j and scores_matrix[i][j]>th):
                count_fnmr+=1
            elif(i!=j and scores_matrix[i][j]<=th):  
                count_fmr+=1  

    FNMR=count_fnmr/len(scores_matrix)
    list_FNMR.append(FNMR)

    FMR=count_fmr/((len(scores_matrix)*len(scores_matrix))-len(scores_matrix))  
    list_FMR.append(FMR)

print("FNMR: ",list_FNMR)
print("FMR: ",list_FMR)
#--------------------------------------------------------------------------------

# Find the index where FNMR equals FMR
eer_index = np.argmin(np.abs(np.array(list_FNMR) - np.array(list_FMR)))

# Determine the Equal Error Rate (EER) threshold
eer_threshold = (list_FNMR[eer_index] + list_FMR[eer_index]) / 2

# Plot ROC curve with EER point
plt.figure(figsize=(8, 6))
plt.plot(list_FNMR, list_FMR, marker='o', linestyle='-', linewidth=2, label='ROC Curve')
plt.plot(list_FNMR[eer_index], list_FMR[eer_index], marker='o', markersize=8, color='red', label='EER Point')
plt.xlabel('False Negative Match Rate (FNMR)')
plt.ylabel('False Match Rate (FMR)')
plt.title('ROC Curve with Equal Error Rate (EER)')
plt.grid(True)
plt.legend()
plt.show()

print("Equal Error Rate (EER) Threshold:", eer_threshold)