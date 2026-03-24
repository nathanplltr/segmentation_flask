import cv2
import numpy as np
import os

def process_image_segmentation(image_path, processed_folder):
    img = cv2.imread(image_path)
    if img is None: return None, []

    filename = os.path.basename(image_path)
    
    # 1. Correction Luminosité (Verrou)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_lum = cv2.cvtColor(cv2.merge((clahe.apply(l),a,b)), cv2.COLOR_LAB2BGR)

    # 2. Segmentation (Masque)
    gray = cv2.cvtColor(img_lum, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Morphologie (Verrou Bruit)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 4. Analyse des Régions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_regions = img.copy()
    regions_list = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > 150: # On filtre les petits bruits
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_regions, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Calcul de forme (Objectif : identifier anomalies)
            ratio = float(w)/h
            label_propose = "Kayak/Paddle" if ratio > 2.5 else "Bateau"
            
            regions_list.append({
                'id': i, 'area': int(area), 'ratio': round(ratio, 2), 'label': label_propose
            })

    # Sauvegardes des images résultantes
    cv2.imwrite(os.path.join(processed_folder, "mask_" + filename), mask)
    cv2.imwrite(os.path.join(processed_folder, "reg_" + filename), img_regions)
    
    return filename, regions_list