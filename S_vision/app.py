import cv2
import torch
import numpy as np
import statistics
from flask import Flask, render_template, Response, jsonify
import threading
from flask_cors import CORS
import warnings
import os
import struct
import snap7
from snap7.util import *
import time
import atexit
import sys
import logging
import requests  # Para enviar datos al servicio de calibración

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,  # Cambia a DEBUG para más detalles
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
sys.stderr = open('snap7_errors.log', 'w')

###############################################################################
#                        IGNORAR LOS FUTUREWARNING DE TORCH                   #
###############################################################################
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################
#                           CONFIGURACIÓN GENERAL                             #
###############################################################################
CONFIDENCE_THRESHOLD = 0.3  # Umbral de confianza para submodelos
IOU_THRESHOLD = 0.3       # Umbral para NMS
CONSENSUS_COUNT = 0         # Confirmar detección inmediatamente
MAX_BUFFER_SIZE = 1         # Tamaño máximo del buffer de lecturas

CAMERA_INDEX = 0          # Índice de la cámara (ajusta según tu sistema)
DETECT_ACTIVE = False
LOCK = threading.Lock()

# Último valor confirmado para evitar envíos repetidos
LAST_CONFIRMED_VALUE = {"medidor": None, "value": None}

# Para formatear el número en Honeywell
initial_dot_pos = 2

# Variable global para almacenar la última imagen válida (para el stream)
LAST_VALID_FRAME = None

###############################################################################
#                 HISTORIAL Y BUFFERS PARA CADA RAMA                           #
###############################################################################
HISTORY = {
    'Medidor_1': [], 
    'Medidor_2': [],
    'Medidor_3': [],
    'Medidor_4': [],
    'Display_digital': [],
    'honeywell': []
}
DETECTION_BUFFER = {
    'Medidor_1': [],
    'Medidor_2': [],
    'Medidor_3': [],
    'Medidor_4': [],
    'Display_digital': [],
    'honeywell': []
}

###############################################################################
#                    FUNCIONES AUXILIARES Y DE VALIDACIÓN                     #
###############################################################################
def load_local_model(pt_file: str):
    """Carga un modelo YOLOv5 local usando torch.hub.load."""
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"El modelo no se encontró en la ruta: {pt_file}")
    return torch.hub.load('./yolov5', 'custom', path=pt_file, source='local', force_reload=False)

def iou(boxA, boxB):
    """Cálculo simple de Intersection Over Union (IOU)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union_area = areaA + areaB - inter_area
    return 0.0 if union_area == 0 else inter_area / union_area

def nms(detections, iou_thres=0.5):
    """Non-Maximum Suppression básico."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    final = []
    while detections:
        best = detections.pop(0)
        final.append(best)
        detections = [d for d in detections if iou(best['box'], d['box']) < iou_thres]
    return final

def check_consensus(buffer_list, required_count=1):
    """Devuelve la última lectura si las últimas 'required_count' lecturas son iguales."""
    if len(buffer_list) < required_count:
        return None
    last_values = buffer_list[-required_count:]
    if all(v == last_values[0] for v in last_values):
        return last_values[-1]
    return None

def is_progressive_valid(medidor, new_str):
    if not HISTORY[medidor]:
        return True
    try:
        old_val = int(HISTORY[medidor][-1])
        new_val = int(new_str)
    except ValueError:
        return False
    # Permite saltos hasta 5 unidades (ajustado)
    return abs(new_val - old_val) <= 5

def is_adaptive_range_valid(medidor, new_str, window=15, k=2):
    if not HISTORY[medidor]:
        return True
    try:
        new_val = int(new_str)
    except ValueError:
        return False
    recent = HISTORY[medidor][-window:]
    if not recent:
        return True
    vals_int = []
    for v in recent:
        try:
            vals_int.append(int(v))
        except ValueError:
            pass
    if not vals_int:
        return True
    media = statistics.mean(vals_int)
    stdev = statistics.pstdev(vals_int)
    if stdev == 0:
        stdev = 1
    rango_min = media - k * stdev
    rango_max = media + k * stdev
    return (rango_min <= new_val <= rango_max)

def is_any_valid(medidor, new_str):
    return is_progressive_valid(medidor, new_str) or is_adaptive_range_valid(medidor, new_str, window=15, k=3)

###############################################################################
#                               CARGA DE MODELOS                               #
###############################################################################
DISPLAY_MODELS_DIR = 'modelos_display'
NUMERIC_MODELS_DIR = 'modelos_numericos'

# Modelos principales para medidores:
display_model = load_local_model(os.path.join(DISPLAY_MODELS_DIR, 'Display.pt'))     # Para Medidor_1, Medidor_2, Medidor_3
display_4_model = load_local_model(os.path.join(DISPLAY_MODELS_DIR, 'Display_4.pt')) # Para Medidor_4

# Submodelos numéricos para medidores:
analogico_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'Analogico.pt'))      # Medidor_1
negro_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'ult.pt'))                # Medidor_1, Medidor_2, Medidor_3
metal_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'metal.pt'))              # Medidor_2
medidor_especial_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'medidor_especial.pt'))  # Medidor_4

MEDIDOR_TO_MODEL = {
    'Medidor_1': analogico_model,
    'Medidor_2': negro_model,
    'Medidor_3': metal_model,
    'Medidor_4': medidor_especial_model
}

# Para Honeywell y Display Digital:
display_honeywell_model = load_local_model(os.path.join(DISPLAY_MODELS_DIR, 'display_honeywell.pt'))
numero_honeywell_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'numero_honeywell.pt'))
subdisplay_honeywell_model = load_local_model(os.path.join(DISPLAY_MODELS_DIR, 'subdisplay_honeywell.pt'))

display_digital_model = load_local_model(os.path.join(DISPLAY_MODELS_DIR, 'Display_digital.pt'))
numero_digital_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'numero_digital.pt'))

###############################################################################
#                             APLICACIÓN FLASK                                #
###############################################################################
app = Flask(__name__)
CORS(app)

###############################################################################
#                        HILO DE CAPTURA DE FRAMES                          #
###############################################################################
class FrameCaptureThread(threading.Thread):
    def __init__(self, camera_index=CAMERA_INDEX):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
    def run(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                logging.error("Error al capturar el frame de la cámara.")
                continue
            with self.lock:
                self.frame = frame
            time.sleep(0.01)
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    def stop(self):
        self.running = False
        self.cap.release()

frame_capture_thread = FrameCaptureThread()
frame_capture_thread.start()

def stop_frame_capture():
    frame_capture_thread.stop()
    print("Hilo de captura de frames detenido.")

atexit.register(stop_frame_capture)

###############################################################################
#              HILO DE DETECCIÓN CONTINUA (MULTI-RAMA)                      #
###############################################################################
class DetectionLoopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.latest_processed_frame = None
    def run(self):
        global DETECT_ACTIVE, LAST_VALID_FRAME
        while self.running:
            frame = frame_capture_thread.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            if DETECT_ACTIVE:
                processed = detect_display(frame)
                combined = combine_rois(processed)
                with LOCK:
                    LAST_VALID_FRAME = combined.copy()
                self.latest_processed_frame = combined
            else:
                self.latest_processed_frame = frame
            time.sleep(0.01)
    def get_processed_frame(self):
        if self.latest_processed_frame is not None:
            return self.latest_processed_frame.copy()
        return None
    def stop(self):
        self.running = False

detection_loop_thread = DetectionLoopThread()
detection_loop_thread.start()

###############################################################################
#              FUNCIÓN DE DETECCIÓN: PROCESAR TODAS LAS RAMAS                #
###############################################################################
def detect_display(frame):
    """
    Procesa las ramas:
      - Medidores (Medidor_1, Medidor_2, Medidor_3 y Medidor_4)
      - Display Digital
      - Honeywell  
    Retorna una lista de ROIs para visualización.
    """
    rois_medidores = []
    rois_digital = []
    rois_honeywell = []
    ih, iw, _ = frame.shape

    # Primero: Verificar la presencia de "rf4" en el frame (usando Display_4.pt)
    rf4_present = False
    results_rf4 = display_4_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_rf4 = results_rf4.xyxyn[0][:, -1].cpu().numpy()
    for label in labels_rf4:
        if display_4_model.names[int(label)] == 'rf4':
            rf4_present = True
            break

    # --- Rama Medidores (Medidor_1, Medidor_2, Medidor_3) ---
    # Solo se procesa esta rama si NO se detectó "rf4" (para evitar confusión con Medidor_4)
    if not rf4_present:
        results_display = display_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        labels_display = results_display.xyxyn[0][:, -1].cpu().numpy()
        boxes_display  = results_display.xyxyn[0][:, :-1].cpu().numpy()
        for label, box in zip(labels_display, boxes_display):
            class_name = display_model.names[int(label)]
            if class_name not in ['Medidor_1', 'Medidor_2', 'Medidor_3']:
                continue
            x1, y1, x2, y2, conf = box
            x1, y1 = int(x1 * iw), int(y1 * ih)
            x2, y2 = int(x2 * iw), int(y2 * ih)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2].copy()
            if roi.size == 0:
                continue
            submodel = MEDIDOR_TO_MODEL.get(class_name)
            if submodel:
                sub_results = submodel(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                dets = []
                has_class_10 = False
                for d in sub_results.xyxy[0]:
                    xa1, ya1, xa2, yb2, conf_sub, cls_sub = d
                    cls_name = submodel.names[int(cls_sub)]
                    if cls_name == '10':
                        has_class_10 = True
                    elif conf_sub >= CONFIDENCE_THRESHOLD:
                        dets.append({
                            'class': cls_name,
                            'box': [int(xa1), int(ya1), int(xa2), int(yb2)],
                            'confidence': float(conf_sub)
                        })
                dets = nms(dets, iou_thres=IOU_THRESHOLD)
                dets_sorted = sorted(dets, key=lambda dd: dd['box'][0])
                new_digits = ''.join(dd['class'] for dd in dets_sorted)
                if has_class_10:
                    cv2.rectangle(roi, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), (255, 0, 0), 2)
                else:
                    if is_any_valid(class_name, new_digits):
                        DETECTION_BUFFER.setdefault(class_name, [])
                        DETECTION_BUFFER[class_name].append(new_digits)
                        if len(DETECTION_BUFFER[class_name]) > MAX_BUFFER_SIZE:
                            DETECTION_BUFFER[class_name].pop(0)
                        confirmed_value = check_consensus(DETECTION_BUFFER[class_name], required_count=CONSENSUS_COUNT)
                        if confirmed_value:
                            try:
                                dato_final = int(confirmed_value)
                            except ValueError:
                                print(f"[ERROR] Valor no válido: {confirmed_value}")
                                continue
                            if LAST_CONFIRMED_VALUE["value"] != dato_final:
                                HISTORY.setdefault(class_name, []).append(dato_final)
                                print(f"[CONSOLE] {class_name} => {dato_final}")
                                with LOCK:
                                    LAST_CONFIRMED_VALUE["medidor"] = class_name
                                    LAST_CONFIRMED_VALUE["value"] = dato_final
                                    DETECTION_BUFFER[class_name].clear()
                                try:
                                    payload = {"dato": dato_final}
                                    respuesta = requests.post("http://localhost:5001/calibration", json=payload, timeout=2)
                                    if respuesta.ok:
                                        print(f"[DEBUG] Valor enviado al servicio de calibración: {dato_final}")
                                    else:
                                        print(f"[ERROR] Error en envío, código: {respuesta.status_code}")
                                except Exception as e:
                                    print(f"[ERROR] Excepción al enviar dato: {e}")
                cv2.putText(roi, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(roi, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), (0, 255, 0), 2)
                rois_medidores.append(roi)

    # --- Rama Medidor_4 ---
    # --- Rama Medidor_4 ---
    results_medidor4 = display_4_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_medidor4 = results_medidor4.xyxyn[0][:, -1].cpu().numpy()
    boxes_medidor4  = results_medidor4.xyxyn[0][:, :-1].cpu().numpy()
    for label, box in zip(labels_medidor4, boxes_medidor4):
        class_name = display_4_model.names[int(label)]
        if class_name != 'Medidor_4':
            continue
        x1, y1, x2, y2, conf = box
        x1, y1 = int(x1 * iw), int(y1 * ih)
        x2, y2 = int(x2 * iw), int(y2 * ih)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue
        submodel = MEDIDOR_TO_MODEL.get('Medidor_4')
        if submodel:
            sub_results = submodel(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            dets = []
            has_class_10 = False
            has_class_11 = False
            for d in sub_results.xyxy[0]:
                xa1, ya1, xa2, yb2, conf_sub, cls_sub = d
                cls_name = submodel.names[int(cls_sub)]
                # Si se detecta la clase "11", se marca y se reemplaza por "6"
                if cls_name == '11':
                    has_class_11 = True
                    cls_name = '6'
                if cls_name == '10':
                    has_class_10 = True
                elif conf_sub >= CONFIDENCE_THRESHOLD:
                    dets.append({
                        'class': cls_name,
                        'box': [int(xa1), int(ya1), int(xa2), int(yb2)],
                        'confidence': float(conf_sub)
                    })
            dets = nms(dets, iou_thres=IOU_THRESHOLD)
            dets_sorted = sorted(dets, key=lambda dd: dd['box'][0])
            new_digits = ''.join(dd['class'] for dd in dets_sorted)
            # Solo se procesa Medidor_4 si se detecta rf4 en el frame o se encontró la clase "11" en el submodelo
            if not (rf4_present or has_class_11):
                # Omitir Medidor_4 para evitar confusión
                continue
            if has_class_10:
                cv2.rectangle(roi, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), (255, 0, 0), 2)
            else:
                if is_any_valid('Medidor_4', new_digits):
                    DETECTION_BUFFER.setdefault('Medidor_4', [])
                    DETECTION_BUFFER['Medidor_4'].append(new_digits)
                    if len(DETECTION_BUFFER['Medidor_4']) > MAX_BUFFER_SIZE:
                        DETECTION_BUFFER['Medidor_4'].pop(0)
                    confirmed_value = check_consensus(DETECTION_BUFFER['Medidor_4'], required_count=CONSENSUS_COUNT)
                    if confirmed_value:
                        try:
                            dato_final = int(confirmed_value)
                        except ValueError:
                            print(f"[ERROR] Valor no válido: {confirmed_value}")
                            continue
                        if LAST_CONFIRMED_VALUE["value"] != dato_final:
                            HISTORY.setdefault('Medidor_4', []).append(dato_final)
                            print(f"[CONSOLE] Medidor_4 => {dato_final}")
                            with LOCK:
                                LAST_CONFIRMED_VALUE["medidor"] = 'Medidor_4'
                                LAST_CONFIRMED_VALUE["value"] = dato_final
                                DETECTION_BUFFER['Medidor_4'].clear()
                            try:
                                payload = {"dato": dato_final}
                                respuesta = requests.post("http://localhost:5001/calibration", json=payload, timeout=2)
                                if respuesta.ok:
                                    print(f"[DEBUG] Valor enviado al servicio de calibración: {dato_final}")
                                else:
                                    print(f"[ERROR] Error en envío, código: {respuesta.status_code}")
                            except Exception as e:
                                print(f"[ERROR] Excepción al enviar dato: {e}")
            cv2.putText(roi, 'Medidor_4', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(roi, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), (0,255,0), 2)
            rois_medidores.append(roi)


    # --- Rama Display Digital ---
    results_digital = display_digital_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_digital = results_digital.xyxyn[0][:, -1].cpu().numpy()
    boxes_digital  = results_digital.xyxyn[0][:, :-1].cpu().numpy()
    for label, box in zip(labels_digital, boxes_digital):
        class_name = display_digital_model.names[int(label)]
        if class_name != 'display_digital':
            continue
        x1, y1, x2, y2, conf = box
        if conf < 0.7:
            continue
        x1, y1 = int(x1 * iw), int(y1 * ih)
        x2, y2 = int(x2 * iw), int(y2 * ih)
        if x2 <= x1 or y2 <= y1:
            continue
        roi_digital = frame[y1:y2, x1:x2].copy()
        if roi_digital.size == 0:
            continue
        results_num_digital = numero_digital_model(cv2.cvtColor(roi_digital, cv2.COLOR_BGR2RGB))
        dets = []
        for d in results_num_digital.xyxy[0]:
            xa1, ya1, xa2, yb2, conf_sub, cls_sub = d
            if conf_sub >= CONFIDENCE_THRESHOLD:
                cls_name_digit = numero_digital_model.names[int(cls_sub)]
                dets.append({
                    'class': cls_name_digit,
                    'box': [int(xa1), int(ya1), int(xa2), int(yb2)],
                    'confidence': float(conf_sub)
                })
        dets = nms(dets, iou_thres=IOU_THRESHOLD)
        dets_sorted = sorted(dets, key=lambda dd: dd['box'][0])
        new_digits = ''.join(dd['class'] for dd in dets_sorted)
        cv2.putText(roi_digital, new_digits, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(roi_digital, (0, 0), (roi_digital.shape[1]-1, roi_digital.shape[0]-1), (0, 255, 0), 2)
        rois_digital.append(roi_digital)
        if is_any_valid('Display_digital', new_digits):
            DETECTION_BUFFER.setdefault('Display_digital', [])
            DETECTION_BUFFER['Display_digital'].append(new_digits)
            if len(DETECTION_BUFFER['Display_digital']) > MAX_BUFFER_SIZE:
                DETECTION_BUFFER['Display_digital'].pop(0)
            confirmed_value = check_consensus(DETECTION_BUFFER['Display_digital'], required_count=CONSENSUS_COUNT)
            if confirmed_value:
                try:
                    dato_final = int(confirmed_value)
                except ValueError:
                    print(f"[ERROR] Valor no válido en Display_digital: {confirmed_value}")
                    continue
                if len(confirmed_value) == 8:
                    if LAST_CONFIRMED_VALUE["value"] != dato_final:
                        HISTORY.setdefault('Display_digital', []).append(dato_final)
                        print(f"[CONSOLE] Display_digital => {dato_final}")
                        with LOCK:
                            LAST_CONFIRMED_VALUE["medidor"] = 'Display_digital'
                            LAST_CONFIRMED_VALUE["value"] = dato_final
                            DETECTION_BUFFER['Display_digital'].clear()
                        try:
                            payload = {"dato": dato_final}
                            respuesta = requests.post("http://localhost:5001/calibration", json=payload, timeout=2)
                            if respuesta.ok:
                                print(f"[DEBUG] Display_digital: dato enviado: {dato_final}")
                            else:
                                print(f"[ERROR] Display_digital: error en envío, código: {respuesta.status_code}")
                        except Exception as e:
                            print(f"[ERROR] Display_digital: excepción al enviar dato: {e}")
                else:
                    print("[DEBUG] Valor en Display_digital no tiene 8 dígitos; se conserva el valor anterior.")

    # --- Rama Honeywell ---
    # Si no se detectaron medidores ni digital, se procesa Honeywell
    if not rois_medidores and not rois_digital:
        results_honeywell = display_honeywell_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        labels_honeywell = results_honeywell.xyxyn[0][:, -1].cpu().numpy()
        boxes_honeywell  = results_honeywell.xyxyn[0][:, :-1].cpu().numpy()
        for label, box in zip(labels_honeywell, boxes_honeywell):
            class_name = display_honeywell_model.names[int(label)]
            if class_name != 'honeywell':
                continue
            x1, y1, x2, y2, conf = box
            x1, y1 = int(x1 * iw), int(y1 * ih)
            x2, y2 = int(x2 * iw), int(y2 * ih)
            if x2 <= x1 or y2 <= y1:
                continue
            roi_honeywell = frame[y1:y2, x1:x2].copy()
            if roi_honeywell.size == 0:
                continue
            roi_honeywell_copia = roi_honeywell.copy()
            sub_results_honeywell = subdisplay_honeywell_model(cv2.cvtColor(roi_honeywell, cv2.COLOR_BGR2RGB))
            dets_subdisplay = []
            has_class_10_honeywell = False
            for d_sub in sub_results_honeywell.xyxy[0]:
                xs1, ys1, xs2, ys2, conf_sub, cls_sub = d_sub
                cls_name_sub = subdisplay_honeywell_model.names[int(cls_sub)]
                if cls_name_sub == '10':
                    has_class_10_honeywell = True
                elif conf_sub >= CONFIDENCE_THRESHOLD:
                    dets_subdisplay.append({
                        'class': cls_name_sub,
                        'box': [int(xs1), int(ys1), int(xs2), int(ys2)],
                        'confidence': float(conf_sub)
                    })
            dets_subdisplay = nms(dets_subdisplay, iou_thres=IOU_THRESHOLD)
            dets_sorted_subdisplay = sorted(dets_subdisplay, key=lambda dd: dd['box'][0])
            for det in dets_sorted_subdisplay:
                x_sub1, y_sub1, x_sub2, y_sub2 = det['box']
                cv2.rectangle(roi_honeywell_copia, (x_sub1, y_sub1), (x_sub2, y_sub2), (255, 255, 0), 2)
                cv2.putText(roi_honeywell_copia, det['class'], (x_sub1, y_sub1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            numero_honeywell_results = numero_honeywell_model(cv2.cvtColor(roi_honeywell, cv2.COLOR_BGR2RGB))
            dets_numero_honeywell = []
            for d_num in numero_honeywell_results.xyxy[0]:
                xn1, yn1, xn2, yn2, conf_num, cls_num = d_num
                if conf_num >= CONFIDENCE_THRESHOLD:
                    cls_name_num = numero_honeywell_model.names[int(cls_num)]
                    dets_numero_honeywell.append({
                        'class': cls_name_num,
                        'box': [int(xn1), int(yn1), int(xn2), int(yn2)],
                        'confidence': float(conf_num)
                    })
            dets_numero_honeywell = nms(dets_numero_honeywell, iou_thres=IOU_THRESHOLD)
            dets_sorted_numero_honeywell = sorted(dets_numero_honeywell, key=lambda dd: dd['box'][0])
            filtered_numbers = []
            for det_num in dets_sorted_numero_honeywell:
                num_box = det_num['box']
                num_center = ((num_box[0] + num_box[2]) / 2, (num_box[1] + num_box[3]) / 2)
                inside = False
                for det_sub in dets_sorted_subdisplay:
                    sub_box = det_sub['box']
                    if (sub_box[0] <= num_center[0] <= sub_box[2]) and (sub_box[1] <= num_center[1] <= sub_box[3]):
                        inside = True
                        break
                if inside:
                    filtered_numbers.append(det_num)
            detected_numbers = [det['class'] for det in filtered_numbers]
            if detected_numbers:
                numero_detectado = ''.join(detected_numbers)
                if numero_detectado:
                    if numero_detectado[0] == '0':
                        pos = initial_dot_pos
                    else:
                        pos = initial_dot_pos + 1
                    if len(numero_detectado) >= pos:
                        numero_formateado = numero_detectado[:pos] + '.' + numero_detectado[pos:]
                    else:
                        numero_formateado = numero_detectado + '.'
                    print(f"[DEBUG] El punto decimal se insertó después del dígito: {pos}")
                    print(f"[DEBUG] Número detectado (con punto): {numero_formateado}")
                else:
                    numero_formateado = numero_detectado
                parts = numero_formateado.split('.')
                if len(parts) >= 2 and len(parts[1]) >= 3:
                    right_part = parts[1][:3]
                    try:
                        dato_detectado = int(right_part)
                    except ValueError:
                        print("[ERROR] No se pudo convertir la parte derecha a entero.")
                        dato_detectado = 0
                    UMBRAL_CAMBIO = 10
                    last_val = LAST_CONFIRMED_VALUE["value"] if LAST_CONFIRMED_VALUE["value"] is not None else dato_detectado
                    if abs(dato_detectado - last_val) > UMBRAL_CAMBIO:
                        print(f"[DEBUG] Cambio abrupto detectado ({dato_detectado} vs {last_val}). Se conserva el valor anterior.")
                    else:
                        if dato_detectado != last_val:
                            print(f"[DEBUG] honeywell => {dato_detectado}")
                            with LOCK:
                                LAST_CONFIRMED_VALUE["medidor"] = 'honeywell'
                                LAST_CONFIRMED_VALUE["value"] = dato_detectado
                            try:
                                payload = {"dato": dato_detectado}
                                respuesta = requests.post("http://localhost:5001/calibration", json=payload, timeout=2)
                                if respuesta.ok:
                                    print(f"[DEBUG] honeywell: dato enviado: {dato_detectado}")
                                else:
                                    print(f"[ERROR] honeywell: error en envío, código: {respuesta.status_code}")
                            except Exception as e:
                                print(f"[ERROR] honeywell: excepción al enviar dato: {e}")
                else:
                    print("[DEBUG] No se detectaron 3 dígitos a la derecha del punto en honeywell, se conserva el valor anterior.")
            cv2.putText(roi_honeywell_copia, 'honeywell', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(roi_honeywell_copia, (0, 0), (roi_honeywell_copia.shape[1]-1, roi_honeywell_copia.shape[0]-1), (0, 255, 0), 2)
            rois_honeywell.append(roi_honeywell_copia)

    if rois_medidores or rois_digital:
        return rois_medidores + rois_digital
    else:
        return rois_honeywell


def combine_rois(rois, max_rois_per_row=4, padding=10):
    """Combina las ROIs en una sola imagen para visualización."""
    if not rois:
        blank = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(blank, 'No detecciones', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return blank
    rois = rois[: max_rois_per_row * 3]
    num_rois = len(rois)
    num_cols = min(num_rois, max_rois_per_row)
    num_rows = (num_rois + max_rois_per_row - 1) // max_rois_per_row
    col_widths = [0] * num_cols
    for i, roi in enumerate(rois):
        c = i % max_rois_per_row
        col_widths[c] = max(col_widths[c], roi.shape[1])
    row_heights = [0] * num_rows
    for i, roi in enumerate(rois):
        r = i // max_rois_per_row
        row_heights[r] = max(row_heights[r], roi.shape[0])
    total_w = sum(col_widths) + padding * (num_cols + 1)
    total_h = sum(row_heights) + padding * (num_rows + 1)
    combined = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    y_off = padding
    idx_roi = 0
    for r in range(num_rows):
        x_off = padding
        for c in range(num_cols):
            if idx_roi >= num_rois:
                break
            roi = rois[idx_roi]
            combined[y_off:y_off+roi.shape[0], x_off:x_off+roi.shape[1]] = roi
            x_off += col_widths[c] + padding
            idx_roi += 1
        y_off += row_heights[r] + padding
    return combined

###############################################################################
#                             ENDPOINTS FLASK                                #
###############################################################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_from_detection_loop(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames_from_detection_loop():
    while True:
        frame = detection_loop_thread.get_processed_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/start', methods=['GET'])
def start_detection():
    global DETECT_ACTIVE
    with LOCK:
        DETECT_ACTIVE = True
    print("[DEBUG] Detección iniciada.")
    return jsonify({'status': 'detección iniciada'})

@app.route('/data', methods=['GET'])
def get_data():
    with LOCK:
        data = LAST_CONFIRMED_VALUE.copy()
    return jsonify(data)

@app.route('/stop', methods=['GET'])
def stop_detection():
    global DETECT_ACTIVE, LAST_CONFIRMED_VALUE
    with LOCK:
        DETECT_ACTIVE = False
        for medidor in DETECTION_BUFFER:
            DETECTION_BUFFER[medidor].clear()
        for medidor in HISTORY:
            HISTORY[medidor].clear()
        LAST_CONFIRMED_VALUE["medidor"] = None
        LAST_CONFIRMED_VALUE["value"] = None
        print("[DEBUG] Detección detenida. LAST_CONFIRMED_VALUE reiniciado a None.")
    return jsonify({'status': 'detección detenida y valores reseteados'})

###############################################################################
#                                 MAIN                                      #
###############################################################################
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("Interrupción recibida. Cerrando aplicación.")
