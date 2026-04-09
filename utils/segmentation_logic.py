import cv2
import numpy as np
import os

def process_image_segmentation(image_path, processed_folder):
    img = cv2.imread(image_path)
    if img is None: return None, []
    filename = os.path.basename(image_path)
    h, w = img.shape[:2]

    # 1. Conversion en HSV (indispensable pour les couleurs vives comme ces points)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Détection de la couleur des points (Rose/Rouge/Orange)
    # On définit deux plages pour le rouge (qui est au début et à la fin de l'échelle Hue)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    color_mask = cv2.add(mask_red1, mask_red2)

    # 3. On crée aussi le masque de l'eau (comme avant) pour servir de filtre
    # Mais cette fois, on s'en sert pour dire : "Cherche les points rouges UNIQUEMENT dans l'eau"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, water_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. INTERSECTION : On garde les points de couleur qui sont dans la zone claire
    final_mask = cv2.bitwise_and(color_mask, water_mask)

    # 5. Dilater un peu les points pour qu'ils soient bien visibles sur le masque
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=1)

    # 6. Extraction des contours pour les cadres verts
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_regions = img.copy()
    regions_list = []
    
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 2: # Très petit seuil car les points sont minuscules
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            
            # On dessine un carré un peu plus large autour du point pour le voir
            margin = 5
            cv2.rectangle(img_regions, (x-margin, y-margin), (x+w_box+margin, y+h_box+margin), (0, 255, 0), 2)
            
            regions_list.append({
                "id": i,
                "label": "Bouée",
                "bbox": [x, y, w_box, h_box]
            })

    # Sauvegardes
    cv2.imwrite(os.path.join(processed_folder, "mask_" + filename), final_mask)
    cv2.imwrite(os.path.join(processed_folder, "reg_" + filename), img_regions)

    return filename, regions_list