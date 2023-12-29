from math import sqrt

import cv2
import numpy as np


def iris_segmentation(img):
    edge_detect(img)
    # pupil = find_pupil(img)
    # if pupil is None:
    #     print("no pupil")
    #     return None
    # else:
    #     iris = find_iris(img, pupil)
    #     if iris is None:
    #         print("no iris")
    #         return None
    #     else:
    #         cv2.circle(img, (pupil[0], pupil[1]), pupil[2], (0, 255, 0), 2)
    #         cv2.circle(img, (iris[0], iris[1]), iris[2], (0, 255, 0), 2)
    #         cv2.imshow("img1", img)
    #         mask = find_eyelids(img, pupil, iris)
    #         return mask


def find_pupil(img):
    # Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise by blurring the image
    img = cv2.medianBlur(img, 9)
    # Increase the constrast
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)

    circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=25,
                               maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            copy = img.copy()
            cv2.circle(copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.imshow("img", copy)

        # Find the smallest circle (assuming it's the pupil)
        min_circle = min(circles[0, :], key=lambda x: x[2])
        return min_circle
    else:
        return None


def find_iris(img, inner_circle):
    # Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("img", img)

    # Find the edges
    edges = cv2.Canny(img, 50, 150)

    cv2.imshow("edges", edges)

    circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=20, param2=5,
                               minRadius=round(inner_circle[2] * 2.1), maxRadius=round(inner_circle[2] * 3))

    if circles is not None:
        print("circles")
        circles = np.uint16(np.around(circles))
        # Find the circle that neighbours the inner circle
        min_circle = min(circles[0, :],
                         key=lambda x: sqrt((x[0] - inner_circle[0]) ** 2 + (x[1] - inner_circle[1]) ** 2))

        return min_circle

    else:
        print("no circles")
        return None


def find_eyelids(img, inner_circle, outer_circle):
    # make the inside of the pupil and outside the iris black
    mask = np.zeros(img.shape, dtype=np.uint8)

    cv2.circle(mask, (outer_circle[0], outer_circle[1]), outer_circle[2], (255, 255, 255), -1)
    cv2.circle(mask, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 0, 0), -1)

    # Apply the mask
    img = cv2.bitwise_and(img, mask)

    # # Find mean intensity of the image
    # mean_intensity = np.mean(img[img > 0])
    #
    # # Make all pixels with intensity much higher than the mean or much lower than the mean black
    # img[(img > mean_intensity * 2) | (img < mean_intensity * 0.2)] = 0

    # blur the image
    img = cv2.medianBlur(img, 5)

    # Increase the constrast
    img = cv2.convertScaleAbs(img, alpha=1.6, beta=30)

    # Find the edges
    edges = cv2.Canny(img, 100, 100)

    # vertical edges or lines like eyelashes are not important for the segmentation of the eyelids
    # so we remove them by dilating and eroding the image
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    cv2.imshow("edges", edges)

    # Find the contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find all the contours that are not the outer circle or the inner circle and are not too small (noise)
    contours = [cnt for cnt in contours if not (outer_circle[0] in cnt and outer_circle[1] in cnt) and not (
            inner_circle[0] in cnt and inner_circle[1] in cnt) and cv2.contourArea(cnt) > 5]
    print([cv2.arcLength(cnt, True) for cnt in contours])

    contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > 30]

    # if all points of the contour are lower than the inner circle, it is the lower eyelid
    # else it is not an eyelid
    upper_eyelid = []
    lower_eyelid = []

    # Find the contour with the largest arc length
    for cnt in contours:
        points = [point[0] for point in cnt]
        # Connect the points with lines
        all_points = []
        for i in range(len(points)):
            all_points.append(points[i])
            all_points.append(points[(i + 1) % len(points)])
            # append all the points between the two points
            all_points.extend([points[(i + 1) % len(points)] + (points[i] - points[(i + 1) % len(points)]) * (j / 10)
                               for j in range(1, 10)])
        points = all_points

        # Check if all points are below the inner circle
        upper = True
        lower = True

        for point in points:
            x = int(point[0])
            y = int(point[1])
            if x >= img.shape[1] or y >= img.shape[0]:
                lower = False
                upper = False

            if y > inner_circle[1] - inner_circle[2] - 10:
                upper = False
            if y < inner_circle[1] + inner_circle[2] + 10:
                lower = False

        if upper:
            upper_eyelid.append(points)
        if lower:
            lower_eyelid.append(points)

    # Find the upper eyelid
    # Assuming upper_eyelid is a list of points (x, y)
    if len(upper_eyelid) > 0:
        points = [p for points in upper_eyelid for p in points]
        all_points = []
        for i in range(len(points)):
            all_points.append(points[i])
            all_points.append(points[(i + 1) % len(points)])
            # append all the points between the two points
            all_points.extend([points[(i + 1) % len(points)] + (points[i] - points[(i + 1) % len(points)]) * (j / 10)
                               for j in range(1, 20)])
        for point in all_points:
            x_upper = int(point[0])
            y_upper = int(point[1])
            cv2.line(mask, (x_upper, y_upper), (x_upper, 0), (0, 0, 0), 5)
            cv2.circle(mask, (x_upper, y_upper), 1, (0, 255, 0), 1)

    # Find the lower eyelid
    # Assuming lower_eyelid is a list of points (x, y)
    if len(lower_eyelid) > 0:
        for points in lower_eyelid:
            all_points = []
            for i in range(len(points)):
                all_points.append(points[i])
                all_points.append(points[(i + 1) % len(points)])
                # append all the points between the two points
                all_points.extend(
                    [points[(i + 1) % len(points)] + (points[i] - points[(i + 1) % len(points)]) * (j / 10)
                     for j in range(1, 20)])
            for point in all_points:
                x_lower = int(point[0])
                y_lower = int(point[1])
                cv2.line(mask, (x_lower, y_lower), (x_lower, img.shape[0]), (0, 0, 0), 2)
                # cv2.circle(mask, (x_lower, y_lower), 1, (0, 0, 25), 1)
    return mask


def edge_detect(img):
    inner = find_pupil(img)
    if inner is None:
        print("no pupil")
        return None
    else:
        outer = find_iris(img, inner)

        if outer is None:
            print("no iris")
            return None
        else:
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.circle(mask, (outer[0], outer[1]), outer[2], (255, 255, 255), -1)
            cv2.circle(mask, (inner[0], inner[1]), inner[2], (0, 0, 0), -1)

            # Apply the mask
            # img = cv2.bitwise_and(img, mask)

    # Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    copy = img.copy()

    # eyelash has (r, g, b) from (60, 60, 60) to (100, 100, 100)
    # blur all pixels with intensity in this range to equal the intensity of the surrounding pixels
    # this will make the eyelash disappear
    eyelash_mask = np.ones(img.shape, dtype=np.uint8)*250
    eyelash_mask[(img > 30) & (img < 110)] = 0

    cv2.imshow("mask", eyelash_mask)

    # Blur the image
    img = cv2.medianBlur(img, 5)

    # Convert the image back to uint8 for display
    img = img.astype(np.uint8)

    edge_mask = cv2.Sobel(eyelash_mask, cv2.CV_64F, 0, 1, ksize=3)
    edge_mask = np.uint8(np.absolute(edge_mask))
    edge_mask = cv2.convertScaleAbs(edge_mask, alpha=1.6, beta=30)

    cv2.imshow("edge_mask", edge_mask)

    # edge detection by sobel
    edges = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.uint8(np.absolute(edges))

    # make the edges high contrast
    edges = cv2.convertScaleAbs(edges, alpha=1.6, beta=30)

    # egdes = 0 where egdes < 100
    edges[edges < 95] = 0

    cv2.imshow("edges", edges)

    # where mask is 0, 3x3 kernel, 1 iteration
    kernel = np.ones((3,3), np.uint8)
    eyelash_mask = cv2.erode(eyelash_mask, kernel, iterations=1)
    edges = cv2.bitwise_and(edges, eyelash_mask)

    # connect the edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    cv2.imshow("edges1", edges)

    # Remove noise from edges using opening
    kernel_opening = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_opening)

    edges = cv2.Canny(edges, 100, 100, apertureSize=5)
    cv2.imshow("edges2", edges)

    # Group the white pixels in the edges together
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # apply mask
    edges = cv2.bitwise_and(edges, eyelash_mask)

    cv2.imshow("edges3", edges)

    # detect contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # remove small contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]

    # draw contours
    cv2.drawContours(copy, contours, -1, (0, 255, 0), 2)
    cv2.imshow("contours", copy)






