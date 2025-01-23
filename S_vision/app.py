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
import queue
import time
import atexit
import sys
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,  # Cambia a DEBUG para más detalles
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Redirigir stderr a un archivo de log
sys.stderr = open('snap7_errors.log', 'w')

###############################################################################
#                        IGNORAR LOS FUTUREWARNING DE TORCH                   #
###############################################################################
warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################
#                           CONFIGURACIÓN GENERAL                             #
###############################################################################
CONFIDENCE_THRESHOLD = 0.5   # Umbral de confianza para submodelos
IOU_THRESHOLD = 0.5          # Umbral para NMS
CONSENSUS_COUNT =  1          # Confirmar detección inmediatamente
MAX_BUFFER_SIZE = 1          # Tamaño máximo del buffer de lecturas

CAMERA_INDEX = 0             # Índice de la cámara (ajusta según tu sistema)
DETECT_ACTIVE = False
LOCK = threading.Lock()

LAST_CONFIRMED_VALUE = {"medidor": None, "value": None}  # Inicializar sin valor confirmado

###############################################################################
#                 HISTORIAL Y BUFFERS PARA CADA MEDIDOR                       #
###############################################################################
# Almacenamos todas las lecturas confirmadas
HISTORY = {
    'Medidor_1': [], 
    'Medidor_2': [],
    'Medidor_3': [],
    'Medidor_4': []
}

# Buffer temporal donde se guardan lecturas hasta que se cumpla el consenso
DETECTION_BUFFER = {
    'Medidor_1': [],
    'Medidor_2': [],
    'Medidor_3': [],
    'Medidor_4': []
}

###############################################################################
#                            FUNCIONES AUXILIARES                             #
###############################################################################
def load_local_model(pt_file: str):
    """
    Carga un modelo YOLOv5 local (sin conexión a internet) desde una ruta específica
    usando torch.hub.load con source='local'.
    """
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"El modelo no se encontró en la ruta: {pt_file}")
    return torch.hub.load(
        './yolov5',  # Asegúrate de que la ruta a yolov5 sea correcta
        'custom',
        path=pt_file,
        source='local',
        force_reload=False
    )

def iou(boxA, boxB):
    """Cálculo simple de Intersection Over Union (IOU)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
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
        detections = [
            d for d in detections 
            if iou(best['box'], d['box']) < iou_thres
        ]
    return final

def check_consensus(buffer_list, required_count=1):
    """
    Verifica si en buffer_list hay un consenso >= required_count.
    - Si todas las últimas 'required_count' lecturas son iguales => confirmamos.
    """
    if len(buffer_list) < required_count:
        return None

    # Tomar solo las últimas 'required_count'
    last_values = buffer_list[-required_count:]

    # Verificar si todas son idénticas
    if all(v == last_values[0] for v in last_values):
        return last_values[-1]  # la última detección se confirma

    return None

###############################################################################
#             FUNCIONES DE VALIDACIÓN (SE ACEPTA SI ALGUNA ES TRUE)           #
###############################################################################
def is_progressive_valid(medidor, new_str):
    """
    Validación 1: Chequeo progresivo (evita saltos muy grandes).
    - Si no hay detecciones confirmadas, se acepta provisionalmente.
    - Si hay una anterior, permite un salto de ±2 (puedes ajustar).
    """
    if not HISTORY[medidor]: 
        return True  # Nada previo => no tenemos con qué comparar

    try:
        old_val = int(HISTORY[medidor][-1])
        new_val = int(new_str)
    except ValueError:
        return False

    # Aquí permitimos ±2, ajusta si deseas más rango
    return abs(new_val - old_val) <= 2

def is_adaptive_range_valid(medidor, new_str, window=10, k=3):
    """
    Validación 2: Rango adaptativo basado en estadísticas del historial.
    - Toma las últimas 'window' lecturas confirmadas
    - Calcula media y desviación estándar
    - Permite new_val dentro de (media ± k*desviación).
    """
    if not HISTORY[medidor]:
        return True  # Si no hay historial, no hay rango => lo aceptamos

    try:
        new_val = int(new_str)
    except ValueError:
        return False

    # Tomar las últimas 'window' lecturas
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
    stdev = statistics.pstdev(vals_int)  # o stdev
    if stdev == 0:
        # Si todas eran iguales, stdev=0 => permitir algo de rango
        stdev = 1

    rango_min = media - k * stdev
    rango_max = media + k * stdev

    return (rango_min <= new_val <= rango_max)

def is_any_valid(medidor, new_str):
    """
    Se acepta si CUALQUIERA de las validaciones (progresiva o rango adaptativo) es True.
    """
    return is_progressive_valid(medidor, new_str) or \
           is_adaptive_range_valid(medidor, new_str, window=10, k=3)

###############################################################################
#                               CARGA DE MODELOS                               #
###############################################################################
# Definir rutas de las carpetas
DISPLAY_MODELS_DIR = 'modelos_display'
NUMERIC_MODELS_DIR = 'modelos_numericos'

# Cargar modelos principales desde 'modelos_display'
display_model = load_local_model(os.path.join(DISPLAY_MODELS_DIR, 'Display.pt'))
display_4_model = load_local_model(os.path.join(DISPLAY_MODELS_DIR, 'Display_4.pt'))

# Cargar submodelos desde 'modelos_numericos'
analogico_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'Analogico.pt'))
negro_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'ult.pt'))  # Asegúrate de que el nombre es correcto
metal_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'metal.pt'))
medidor_especial_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'medidor_especial.pt'))  # Submodelo para Medidor_4

# Asociar modelos con medidores
MEDIDOR_TO_MODEL = {
    'Medidor_1': analogico_model,
    'Medidor_2': negro_model,
    'Medidor_3': metal_model,
    'Medidor_4': medidor_especial_model  # Asociamos Medidor_4 con su submodelo especial
}

###############################################################################
#                             CONEXIÓN CON EL PLC                              #
###############################################################################
# Crear una cola para almacenar los datos a enviar al PLC
plc_queue = queue.Queue()

# Configuración del PLC
PLC_IP = '192.168.0.12'
PLC_RACK = 0
PLC_SLOT = 1

# Inicializar el cliente snap7
cliente = snap7.client.Client()

# Variable para controlar la impresión del mensaje de espera
print_waiting_message = False

def plc_worker():
    global cliente, print_waiting_message
    while True:
        if not cliente.get_connected():
            try:
                cliente.connect(PLC_IP, PLC_RACK, PLC_SLOT)
                if cliente.get_connected():
                    print(f"Conectado al PLC en {PLC_IP}, rack {PLC_RACK}, slot {PLC_SLOT}")
                    print_waiting_message = False  # Resetear la bandera
            except Exception as e:
                if not print_waiting_message:
                    print("Esperando conexión con el PLC...")
                    print_waiting_message = True
                time.sleep(5)  # Esperar 5 segundos antes de reintentar
                continue  # Reintentar la conexión

        try:
            # Intentar obtener un dato de la cola con timeout reducido
            dato_final = plc_queue.get(timeout=0.1)
            # Empaquetar el dato como un entero de 4 bytes en Big Endian
            data_word = bytearray(struct.pack('>I', dato_final))
            db_number = 1
            start = 0
            cliente.db_write(db_number, start, data_word)
            print(f"Escritura en DB {db_number}: {dato_final}")
        except queue.Empty:
            continue  # No hay datos para enviar
        except Exception as e:
            print(f"Error al escribir en el PLC: {e}")
            cliente.disconnect()
            print_waiting_message = True  # Asegurar que el mensaje de espera se imprima

# Iniciar el hilo del PLC
plc_thread = threading.Thread(target=plc_worker, daemon=True)
plc_thread.start()

# Asegurar que la conexión al PLC se cierre al finalizar la aplicación
def close_plc_connection():
    if cliente.get_connected():
        cliente.disconnect()
        print("Desconectado del PLC.")

atexit.register(close_plc_connection)

###############################################################################
#                             APLICACIÓN FLASK                                #
###############################################################################
app = Flask(__name__)
CORS(app)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reducir resolución para mayor velocidad
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Configurar FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/start', methods=['GET'])
def start_detection():
    global DETECT_ACTIVE
    with LOCK:
        DETECT_ACTIVE = True
    print("[DEBUG] Detección iniciada.")
    return jsonify({'status': 'detección iniciada'})

@app.route('/data', methods=['GET'])
def get_data():
    """
    Endpoint para obtener el último valor confirmado.
    Retorna un JSON con el nombre del medidor y el valor detectado.
    """
    global LAST_CONFIRMED_VALUE
    with LOCK:
        data = LAST_CONFIRMED_VALUE.copy()
    return jsonify(data)

@app.route('/stop', methods=['GET'])
def stop_detection():
    """
    Al detener la detección:
    - Se limpia HISTORY, DETECTION_BUFFER
    - Se pone DETECT_ACTIVE en False
    - Se reinicia LAST_CONFIRMED_VALUE a valores por defecto
    """
    global DETECT_ACTIVE, LAST_CONFIRMED_VALUE
    with LOCK:
        DETECT_ACTIVE = False
        # Resetear buffers e historial
        for medidor in DETECTION_BUFFER:
            DETECTION_BUFFER[medidor].clear()
        for medidor in HISTORY:
            HISTORY[medidor].clear()
        # Reiniciar LAST_CONFIRMED_VALUE fuera de los bucles
        LAST_CONFIRMED_VALUE["medidor"] = None
        LAST_CONFIRMED_VALUE["value"] = None  # Asigna a None para indicar ausencia de valor
        print("[DEBUG] Detección detenida. LAST_CONFIRMED_VALUE reiniciado a None.")
    return jsonify({'status': 'detección detenida y valores reseteados'})

def generate_frames():
    global DETECT_ACTIVE
    while True:
        success, frame = cap.read()
        if not success:
            break
        with LOCK:
            active = DETECT_ACTIVE

        if active:
            rois = detect_display(frame)
            combined = combine_rois(rois)
        else:
            combined = frame  # Devuelve directamente el frame original

        ret, buffer = cv2.imencode('.jpg', combined)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

###############################################################################
#                           FUNCIÓN PRINCIPAL DE DETECCIÓN                    #
###############################################################################
def detect_display(frame):
    """
    Función de detección de medidores y manejo de ROIs.
    """
    rois = []
    ih, iw, _ = frame.shape

    # Paso 1: Detectar rf4 usando Display_4.pt
    results_rf4 = display_4_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_rf4 = results_rf4.xyxyn[0][:, -1].cpu().numpy()
    boxes_rf4  = results_rf4.xyxyn[0][:, :-1].cpu().numpy()

    rf4_present = False
    for label, box in zip(labels_rf4, boxes_rf4):
        class_id = int(label)
        class_name = display_4_model.names[class_id]
        if class_name == 'rf4':
            rf4_present = True
            break  # Solo necesita al menos un rf4

    # Paso 2: Detectar Medidor_4 usando Display_4.pt
    results_medidor_4 = display_4_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_medidor_4 = results_medidor_4.xyxyn[0][:, -1].cpu().numpy()
    boxes_medidor_4  = results_medidor_4.xyxyn[0][:, :-1].cpu().numpy()

    medidor_4_rois = []

    if rf4_present:
        for label, box in zip(labels_medidor_4, boxes_medidor_4):
            class_id = int(label)
            class_name = display_4_model.names[class_id]
            if class_name != 'Medidor_4':
                continue  # Solo procesar Medidor_4

            x1, y1, x2, y2, conf = box
            x1, y1 = int(x1 * iw), int(y1 * ih)
            x2, y2 = int(x2 * iw), int(y2 * ih)

            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2].copy()
            if roi.size == 0:
                continue

            # Aplicar el submodelo especial para Medidor_4
            submodel = MEDIDOR_TO_MODEL.get('Medidor_4')
            if not submodel:
                continue

            sub_results = submodel(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            dets = []
            has_class_10 = False  # Flag para detectar si hay clase '10'

            for d in sub_results.xyxy[0]:
                xa1, ya1, xa2, yb2, conf_sub, cls_sub = d
                cls_name = submodel.names[int(cls_sub)]
                # Convertir clase 11 a '6'
                cls_name = '6' if cls_name == '11' else cls_name
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
                # Dibujar ROI y etiqueta indicando que se detectó '10'
                cv2.rectangle(roi, (0,0), (roi.shape[1]-1, roi.shape[0]-1), (255,0,0), 2)  # Rojo para indicar problema
                cv2.putText(roi, 'Clase 10 detectada', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                rois.append(roi)  # Agregar la ROI sin añadir al buffer ni historial
                # Guardar la ROI para excluirla en Display.pt
                medidor_4_rois.append([x1, y1, x2, y2])
                continue  # No agregar al buffer ni al historial
            else:
                # Validar y confirmar
                if is_any_valid('Medidor_4', new_digits):
                    DETECTION_BUFFER['Medidor_4'].append(new_digits)
                    if len(DETECTION_BUFFER['Medidor_4']) > MAX_BUFFER_SIZE:
                        DETECTION_BUFFER['Medidor_4'].pop(0)

                    confirmed_value = check_consensus(
                        DETECTION_BUFFER['Medidor_4'],
                        required_count=CONSENSUS_COUNT
                    )
                    if confirmed_value:
                        try:
                            dato_final = int(confirmed_value)  # Convertir a entero
                        except ValueError:
                            print(f"[ERROR] Valor no válido para convertir a entero: {confirmed_value}")
                            continue  # O maneja el error según tus necesidades

                        HISTORY['Medidor_4'].append(dato_final)
                        print(f"[CONSOLE] Medidor_4 => {dato_final}")
                        with LOCK:
                            if DETECT_ACTIVE:
                                LAST_CONFIRMED_VALUE["medidor"] = 'Medidor_4'
                                LAST_CONFIRMED_VALUE["value"] = dato_final
                                DETECTION_BUFFER['Medidor_4'].clear()

                                # Encolar el dato_final para enviarlo al PLC
                                try:
                                    plc_queue.put_nowait(dato_final)  # Ya es entero
                                    print(f"[DEBUG] Valor encolado para el PLC: {dato_final}")
                                except queue.Full:
                                    print("La cola para el PLC está llena. Se omitirá el dato.")
                            else:
                                print("[DEBUG] Detección detenida.")
    
            # Dibujar ROI y etiqueta
            cv2.putText(roi, 'Medidor_4', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(roi, (0,0), (roi.shape[1]-1, roi.shape[0]-1), (0,255,0), 2)
            rois.append(roi)

            # Guardar la ROI para excluirla en Display.pt
            medidor_4_rois.append([x1, y1, x2, y2])

    # Paso 3: Detectar Medidor_1, Medidor_2, Medidor_3 usando Display.pt
    results_display = display_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_display = results_display.xyxyn[0][:, -1].cpu().numpy()
    boxes_display  = results_display.xyxyn[0][:, :-1].cpu().numpy()

    for label, box in zip(labels_display, boxes_display):
        class_id = int(label)
        class_name = display_model.names[class_id]
        if class_name not in ['Medidor_1', 'Medidor_2', 'Medidor_3']:
            continue  # Solo procesar estos medidores

        x1, y1, x2, y2, conf = box
        x1, y1 = int(x1 * iw), int(y1 * ih)
        x2, y2 = int(x2 * iw), int(y2 * ih)

        if x2 <= x1 or y2 <= y1:
            continue

        # Verificar si esta ROI se solapa con alguna de las ROIs de Medidor_4
        overlap = False
        for med4_box in medidor_4_rois:
            iou_val = iou([x1, y1, x2, y2], med4_box)
            if iou_val > 0.3:  # Ajusta el umbral según sea necesario
                overlap = True
                break
        if overlap:
            continue  # Excluir esta detección, ya está manejada por Medidor_4

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        # Aplicar el submodelo correspondiente
        submodel = MEDIDOR_TO_MODEL.get(class_name)
        if submodel:
            # Detectar dígitos con el submodelo
            sub_results = submodel(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            dets = []
            has_class_10 = False  # Flag para detectar si hay clase '10'

            for d in sub_results.xyxy[0]:
                xa1, ya1, xa2, yb2, conf_sub, cls_sub = d
                cls_name = submodel.names[int(cls_sub)]
                if class_name == 'Medidor_4':
                    # En Medidor_4 ya hemos manejado la clase '10' y la conversión de '11' a '6' arriba
                    continue
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

            if class_name == 'Medidor_2':
                if has_class_10:
                    # Dibujar ROI y etiqueta indicando que se detectó '10'
                    cv2.rectangle(roi, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), (255, 0, 0), 2)  # Rojo para indicar problema
                    cv2.putText(roi, 'Clase 10 detectada', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    rois.append(roi)  # Agregar la ROI sin añadir al buffer ni historial

                    # Verificar si el último valor confirmado es de Medidor_2
                    with LOCK:
                        if LAST_CONFIRMED_VALUE["medidor"] == 'Medidor_2' and LAST_CONFIRMED_VALUE["value"] is not None:
                            print(f"[CONSOLE] Medidor_2 => {LAST_CONFIRMED_VALUE['value']}")
                        else:
                            print("[CONSOLE] Medidor_2 => Esperando detección válida.")
                    continue  # No agregar al buffer ni al historial
                else:
                    # Si no se detecta '10', proceder normalmente
                    print(f"[CONSOLE] Medidor_2 => {new_digits}")

            # --- NUEVO BLOQUE PARA MEDIDOR_3 ---
            if class_name == 'Medidor_3':
                if has_class_10:
                    # Dibujar ROI y etiqueta indicando que se detectó '10' (o la clase específica)
                    cv2.rectangle(roi, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), (255, 0, 0), 2)  # Rojo para indicar problema
                    cv2.putText(roi, 'Clase 10 detectada', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    rois.append(roi)  # Agregar la ROI sin añadir al buffer ni historial

                    # Verificar si el último valor confirmado es de Medidor_3
                    with LOCK:
                        if LAST_CONFIRMED_VALUE["medidor"] == 'Medidor_3' and LAST_CONFIRMED_VALUE["value"] is not None:
                            print(f"[CONSOLE] Medidor_3 => {LAST_CONFIRMED_VALUE['value']}")
                        else:
                            print("[CONSOLE] Medidor_3 => Esperando detección válida.")
                    continue  # No agregar al buffer ni al historial
                else:
                    # Si no se detecta '10', proceder normalmente
                    print(f"[CONSOLE] Medidor_3 => {new_digits}")
            # --- FIN BLOQUE PARA MEDIDOR_3 ---

            # Manejar detecciones de clase '10' para otros medidores si es necesario
            if has_class_10 and class_name not in ['Medidor_2', 'Medidor_3']:
                # Dibujar ROI y etiqueta indicando que se detectó '10'
                cv2.rectangle(roi, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), (255, 0, 0), 2)  # Rojo para indicar problema
                cv2.putText(roi, 'Clase 10 detectada', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                rois.append(roi)  # Agregar la ROI sin añadir al buffer ni historial
                continue  # No agregar al buffer ni al historial
            else:
                # Validar y confirmar
                if is_any_valid(class_name, new_digits):
                    DETECTION_BUFFER[class_name].append(new_digits)
                    if len(DETECTION_BUFFER[class_name]) > MAX_BUFFER_SIZE:
                        DETECTION_BUFFER[class_name].pop(0)

                    confirmed_value = check_consensus(
                        DETECTION_BUFFER[class_name],
                        required_count=CONSENSUS_COUNT
                    )
                    if confirmed_value:
                        try:
                            dato_final = int(confirmed_value)  # Convertir a entero
                        except ValueError:
                            print(f"[ERROR] Valor no válido para convertir a entero: {confirmed_value}")
                            continue  # O maneja el error según tus necesidades

                        HISTORY[class_name].append(dato_final)
                        print(f"[CONSOLE] {class_name} => {dato_final}")
                        with LOCK:
                            if DETECT_ACTIVE:
                                LAST_CONFIRMED_VALUE["medidor"] = class_name
                                LAST_CONFIRMED_VALUE["value"] = dato_final
                                DETECTION_BUFFER[class_name].clear()

                                # Encolar el dato_final para enviarlo al PLC
                                try:
                                    plc_queue.put_nowait(dato_final)  # Ya es entero
                                    print(f"[DEBUG] Valor encolado para el PLC: {dato_final}")
                                except queue.Full:
                                    print("La cola para el PLC está llena. Se omitirá el dato.")
                            else:
                                print("[DEBUG] Detección detenida. No se actualiza LAST_CONFIRMED_VALUE.")

        # Dibujar ROI y etiqueta
        if has_class_10 and class_name in ['Medidor_2', 'Medidor_3']:
            cv2.putText(roi, 'Clase 10 detectada', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(roi, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(roi, (0, 0), (roi.shape[1]-1, roi.shape[0]-1), (0, 255, 0), 2)
        rois.append(roi)

    return rois

def combine_rois(rois, max_rois_per_row=4, padding=10):
    """Combina múltiples ROIs en una sola imagen sin redimensionarlas."""
    if not rois:
        blank = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(blank, 'No detecciones', (50,150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return blank

    rois = rois[: max_rois_per_row * 3]
    num_rois = len(rois)
    num_cols = min(num_rois, max_rois_per_row)
    num_rows = (num_rois + max_rois_per_row - 1) // max_rois_per_row

    widths  = [r.shape[1] for r in rois]
    heights = [r.shape[0] for r in rois]

    col_widths = [0]*num_cols
    for i, roi in enumerate(rois):
        c = i % max_rois_per_row
        col_widths[c] = max(col_widths[c], roi.shape[1])

    row_heights = [0]*num_rows
    for i, roi in enumerate(rois):
        r = i // max_rois_per_row
        row_heights[r] = max(row_heights[r], roi.shape[0])

    total_w = sum(col_widths) + padding*(num_cols+1)
    total_h = sum(row_heights) + padding*(num_rows+1)

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

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    # Iniciar el hilo del PLC antes de correr Flask
    plc_thread = threading.Thread(target=plc_worker, daemon=True)
    plc_thread.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)
 