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
IOU_THRESHOLD = 0.3          # Umbral para NMS
CONSENSUS_COUNT =  1         # Confirmar detección inmediatamente
MAX_BUFFER_SIZE = 1          # Tamaño máximo del buffer de lecturas

CAMERA_INDEX = 0             # Índice de la cámara (ajusta según tu sistema)
DETECT_ACTIVE = False
LOCK = threading.Lock()

LAST_CONFIRMED_VALUE = {"medidor": None, "value": None}  # Inicializar sin valor confirmado

###############################################################################
#                 HISTORIAL Y BUFFERS PARA CADA MEDIDOR                       #
###############################################################################
HISTORY = {
    'Medidor_1': [], 
    'Medidor_2': [],
    'Medidor_3': [],
    'Medidor_4': []
}

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
    - Si las últimas 'required_count' lecturas son iguales, se confirma.
    """
    if len(buffer_list) < required_count:
        return None

    last_values = buffer_list[-required_count:]
    if all(v == last_values[0] for v in last_values):
        return last_values[-1]
    return None

def is_progressive_valid(medidor, new_str):
    """
    Validación 1: Chequeo progresivo (evita saltos muy grandes).
    """
    if not HISTORY[medidor]: 
        return True  # Sin historial previo
    try:
        old_val = int(HISTORY[medidor][-1])
        new_val = int(new_str)
    except ValueError:
        return False
    return abs(new_val - old_val) <= 20

def is_adaptive_range_valid(medidor, new_str, window=15, k=2):
    """
    Validación 2: Rango adaptativo basado en estadísticas del historial.
    """
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
    """
    Se acepta si alguna de las validaciones (progresiva o rango adaptativo) es verdadera.
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
negro_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'ult.pt'))
metal_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'metal.pt'))
medidor_especial_model = load_local_model(os.path.join(NUMERIC_MODELS_DIR, 'medidor_especial.pt'))

# Asociar modelos con medidores
MEDIDOR_TO_MODEL = {
    'Medidor_1': analogico_model,
    'Medidor_2': negro_model,
    'Medidor_3': metal_model,
    'Medidor_4': medidor_especial_model
}

###############################################################################
#                             CONEXIÓN CON EL PLC                              #
###############################################################################
plc_queue = queue.Queue()
PLC_IP = '172.16.181.10'
PLC_RACK = 0
PLC_SLOT = 2
cliente = snap7.client.Client()
print_waiting_message = False

def plc_worker():
    global cliente, print_waiting_message
    while True:
        if not cliente.get_connected():
            try:
                cliente.connect(PLC_IP, PLC_RACK, PLC_SLOT)
                if cliente.get_connected():
                    print(f"Conectado al PLC en {PLC_IP}, rack {PLC_RACK}, slot {PLC_SLOT}")
                    print_waiting_message = False
            except Exception as e:
                if not print_waiting_message:
                    print("Esperando conexión con el PLC...")
                    print_waiting_message = True
                time.sleep(5)
                continue

        try:
            dato_final = plc_queue.get(timeout=0.0)
            data_word = bytearray(struct.pack('>I', dato_final))
            db_number = 10
            start = 4
            cliente.db_write(db_number, start, data_word)
            print(f"Escritura en DB {db_number}: {dato_final}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error al escribir en el PLC: {e}")
            cliente.disconnect()
            print_waiting_message = True

plc_thread = threading.Thread(target=plc_worker, daemon=True)
plc_thread.start()

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

###############################################################################
#                           HILO DE CAPTURA DE FRAMES                         #
###############################################################################
class FrameCaptureThread(threading.Thread):
    def __init__(self, camera_index=CAMERA_INDEX):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
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
#                          CLASE PARA DETECCIÓN POR MEDIDOR                   #
###############################################################################
class DetectionThread(threading.Thread):
    def __init__(self, medidor_name, submodel):
        super().__init__(daemon=True)
        self.medidor_name = medidor_name
        self.submodel = submodel
        self.queue = queue.Queue()
        self.processed_rois = queue.Queue()

    def run(self):
        while True:
            try:
                frame, roi, box = self.queue.get()
                if frame is None:
                    break
                self.process_roi(frame, roi, box)
            except Exception as e:
                logging.error(f"Error en hilo {self.medidor_name}: {e}")

    def process_roi(self, frame, roi, box):
        global LAST_CONFIRMED_VALUE
        ih, iw, _ = frame.shape
        sub_results = self.submodel(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        dets = []
        has_class_10 = False

        for d in sub_results.xyxy[0]:
            xa1, ya1, xa2, yb2, conf_sub, cls_sub = d
            cls_name = self.submodel.names[int(cls_sub)]
            if self.medidor_name == 'Medidor_4' and cls_name == '11':
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

        if has_class_10:
            cv2.rectangle(roi, (0,0), (roi.shape[1]-1, roi.shape[0]-1), (255,0,0), 2)
            self.processed_rois.put(roi)
            return
        else:
            if is_any_valid(self.medidor_name, new_digits):
                with LOCK:
                    DETECTION_BUFFER[self.medidor_name].append(new_digits)
                    if len(DETECTION_BUFFER[self.medidor_name]) > MAX_BUFFER_SIZE:
                        DETECTION_BUFFER[self.medidor_name].pop(0)
                confirmed_value = check_consensus(
                    DETECTION_BUFFER[self.medidor_name],
                    required_count=CONSENSUS_COUNT
                )
                if confirmed_value:
                    try:
                        dato_final = int(confirmed_value)
                    except ValueError:
                        logging.error(f"Valor no válido para convertir a entero: {confirmed_value}")
                        return
                    with LOCK:
                        HISTORY[self.medidor_name].append(dato_final)
                        print(f"[CONSOLE] {self.medidor_name} => {dato_final}")
                        if DETECT_ACTIVE:
                            LAST_CONFIRMED_VALUE["medidor"] = self.medidor_name
                            LAST_CONFIRMED_VALUE["value"] = dato_final
                            DETECTION_BUFFER[self.medidor_name].clear()
                            try:
                                plc_queue.put_nowait(dato_final)
                                print(f"[DEBUG] Valor encolado para el PLC: {dato_final}")
                            except queue.Full:
                                print("La cola para el PLC está llena. Se omitirá el dato.")
                        else:
                            print("[DEBUG] Detección detenida.")

        cv2.rectangle(roi, (0,0), (roi.shape[1]-1, roi.shape[0]-1), (0,255,0), 2)
        self.processed_rois.put(roi)

# Crear y arrancar hilos de detección para cada medidor
detection_threads = {}
for medidor, model in MEDIDOR_TO_MODEL.items():
    thread = DetectionThread(medidor, model)
    thread.start()
    detection_threads[medidor] = thread

###############################################################################
#                     HILO INDEPENDIENTE DE DETECCIÓN                         #
###############################################################################
class DetectionLoopThread(threading.Thread):
    """
    Hilo que obtiene frames de la cámara y, según el estado de DETECT_ACTIVE,
    procesa la detección y guarda el último frame procesado.
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.latest_processed_frame = None

    def run(self):
        global DETECT_ACTIVE
        while self.running:
            frame = frame_capture_thread.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if DETECT_ACTIVE:
                rois = detect_display(frame)
                combined = combine_rois(rois)
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
#                           FUNCIÓN DE DETECCIÓN                              #
###############################################################################
def detect_display(frame):
    """
    Función que detecta medidores y gestiona las ROIs.
    """
    rois = []
    ih, iw, _ = frame.shape

    # Paso 1: Detectar rf4 con Display_4.pt
    results_rf4 = display_4_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_rf4 = results_rf4.xyxyn[0][:, -1].cpu().numpy()
    boxes_rf4  = results_rf4.xyxyn[0][:, :-1].cpu().numpy()

    rf4_present = False
    for label, box in zip(labels_rf4, boxes_rf4):
        class_id = int(label)
        class_name = display_4_model.names[class_id]
        if class_name == 'rf4':
            rf4_present = True
            break

    # Paso 2: Detectar Medidor_4
    results_medidor_4 = display_4_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_medidor_4 = results_medidor_4.xyxyn[0][:, -1].cpu().numpy()
    boxes_medidor_4  = results_medidor_4.xyxyn[0][:, :-1].cpu().numpy()
    medidor_4_rois = []

    if rf4_present:
        for label, box in zip(labels_medidor_4, boxes_medidor_4):
            class_id = int(label)
            class_name = display_4_model.names[class_id]
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

            detection_threads['Medidor_4'].queue.put((frame, roi, box))
            medidor_4_rois.append([x1, y1, x2, y2])

    # Paso 3: Detectar Medidor_1, Medidor_2, Medidor_3
    results_display = display_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    labels_display = results_display.xyxyn[0][:, -1].cpu().numpy()
    boxes_display  = results_display.xyxyn[0][:, :-1].cpu().numpy()

    for label, box in zip(labels_display, boxes_display):
        class_id = int(label)
        class_name = display_model.names[class_id]
        if class_name not in ['Medidor_1', 'Medidor_2', 'Medidor_3']:
            continue

        x1, y1, x2, y2, conf = box
        x1, y1 = int(x1 * iw), int(y1 * ih)
        x2, y2 = int(x2 * iw), int(y2 * ih)

        if x2 <= x1 or y2 <= y1:
            continue

        overlap = False
        for med4_box in medidor_4_rois:
            if iou([x1, y1, x2, y2], med4_box) > 0.3:
                overlap = True
                break
        if overlap:
            continue

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        if class_name in detection_threads:
            detection_threads[class_name].queue.put((frame, roi, box))

    # Recolectar ROIs procesadas
    for medidor, thread in detection_threads.items():
        while not thread.processed_rois.empty():
            processed_roi = thread.processed_rois.get()
            rois.append(processed_roi)

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
    return Response(
        generate_frames_from_detection_loop(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

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
    global LAST_CONFIRMED_VALUE
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
#                              MAIN                                          #
###############################################################################
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("Interrupción recibida. Cerrando aplicación.")
