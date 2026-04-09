import cv2
import numpy as np
import os
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

# Chemin vers le modèle YOLOv8 personnalisé (après entraînement)
YOLO_MODEL_PATH = "yolov8n.pt"  # petit modèle rapide, télécharge automatiquement si absent
yolo_model = YOLO(YOLO_MODEL_PATH)

# Classes maritimes dans le modèle YOLO
MARITIME_CLASSES = ["Voilier", "Kayak/Paddle", "Bateau moteur", "Bouée"]

# Seuil minimal de confiance pour garder une détection
CONF_THRESHOLD = 0.5

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

# Charge le modèle YOLOv8
yolo_model = YOLO(YOLO_MODEL_PATH)


def _filter_herbe(roi_bgr):
    """
    Détecte l'herbe dans une zone et retourne True si la zone est probablement de l'herbe.
    """
    mean_b, mean_g, mean_r = np.mean(roi_bgr.reshape(-1, 3), axis=0)
    return mean_g > mean_r * 1.2 and mean_g > mean_b * 1.2


def _generate_object_masks(contours, shape, processed_folder, filename, img):
    """
    Génère et sauvegarde un masque binaire + un crop coloré pour chaque objet détecté.
    """
    h_img, w_img = shape[:2]
    individual_masks = []

    for i, cnt in enumerate(contours):
        obj_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(obj_mask, [cnt], -1, 255, -1)

        x, y, w, h = cv2.boundingRect(cnt)
        obj_crop = cv2.bitwise_and(img, img, mask=obj_mask)
        obj_crop_roi = obj_crop[y:y+h, x:x+w]

        mask_path = os.path.join(processed_folder, f"obj_{i}_mask_{filename}")
        crop_path = os.path.join(processed_folder, f"obj_{i}_crop_{filename}")
        cv2.imwrite(mask_path, obj_mask)
        cv2.imwrite(crop_path, obj_crop_roi)

        individual_masks.append({"id": i, "mask_path": mask_path, "crop_path": crop_path})

    return individual_masks


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def process_image_segmentation(image_path, processed_folder):
    """
    Pipeline de détection d'objets maritimes via YOLOv8.
    Retourne : filename, regions_list (infos objets détectés)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, []

    filename = os.path.basename(image_path)
    h_img, w_img = img.shape[:2]

    # --- Détection YOLO ---
    results = yolo_model(img)[0]

    img_regions = img.copy()
    regions_list = []
    contours_for_masks = []

    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2]
        conf = float(box.conf.cpu().numpy())
        cls_id = int(box.cls.cpu().numpy())
        class_name = MARITIME_CLASSES[cls_id] if cls_id < len(MARITIME_CLASSES) else "Objet maritime"

        if conf < CONF_THRESHOLD:
            continue  # Ignore faible confiance

        x1, y1, x2, y2 = map(int, xyxy)
        roi_bgr = img[y1:y2, x1:x2]
        if _filter_herbe(roi_bgr):
            continue  # Ignore herbe ou zones vertes parasites

        # Dessine le rectangle et label
        cv2.rectangle(img_regions, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_regions, class_name, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Crée un contour pour le masque
        contour = np.array([
            [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
        ])
        contours_for_masks.append(contour)

        regions_list.append({
            "id": len(regions_list),
            "label": class_name,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
        })

    # --- Génération des masques individuels ---
    _generate_object_masks(contours_for_masks, img.shape,
                           processed_folder, filename, img)

    # --- Sauvegarde de l'image annotée ---
    cv2.imwrite(os.path.join(processed_folder, "reg_" + filename), img_regions)

    return filename, regions_list