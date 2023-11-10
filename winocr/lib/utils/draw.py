import cv2 

def draw(img, boxes=None, 
         color=(0, 255, 0), thickness=2):
    """
    Draw image
    """
    debug_img = img.copy()
    if boxes is not None:
        for box in boxes:
            cv2.rectangle(debug_img, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        color=color, 
                        thickness=thickness)
    return debug_img