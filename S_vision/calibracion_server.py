import snap7
from snap7.util import get_bool, set_bool
import struct
import csv
import time
import logging
import threading
from flask import Flask, request, jsonify
import atexit
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ------------------------------------------------------------------------------
#                   CONFIGURACIÓN DE PARAMETROS GENERALES
# ------------------------------------------------------------------------------
PLC_IP = '192.168.0.10'
PLC_RACK = 0
PLC_SLOT = 1




# Hasta cuántas repeticiones manejar (1 a 10)
NUM_REPETICIONES = 10
print('TOMA TUS REPETICIONES')
  # Se implementan hasta 10 repeticiones
# Incremento (litro) que marca el inicio de cada repetición
GAP = 2  
# Nombre del test (opcional, para logs)
TEST = "calentamiento"

# ------------------------------------------------------------------------------
#                   CONFIGURACIÓN DE DB PARA BANDERA / LECTURAS
# ------------------------------------------------------------------------------
FLAG_DB = 25
FLAG_OFFSET = 20  # DBX8.0 (bool)



PATTERN_DB = 25
PATTERN_OFFSET = 16  # Se lee como FLOAT

SETPOINT_DB = 3
SETPOINT_OFFSET = 22  # Se lee como INT

# ------------------------------------------------------------------------------
#                   CONFIGURACIÓN DE SALIDA A CSV
# ------------------------------------------------------------------------------
CSV_FILE = "calibracion.csv"
CSV_HEADER = ["Evento", "Tipo", "Valor", "Valor_Inicial", "Diferencia", "Setpoint", "Timestamp"]

# ------------------------------------------------------------------------------
#                   DICCIONARIOS PARA ESCRIBIR EN EL PLC (PARÁMETRICO)
# ------------------------------------------------------------------------------
CAMERA_DB_MAP = {
    "MR1I":  {"db": 31, "offset": 128,   "type": "float"},
    "MR1F":  {"db": 31, "offset": 132,  "type": "float"},
    "MR2I":  {"db": 31, "offset": 136,  "type": "float"},
    "MR2F":  {"db": 31, "offset": 140,  "type": "float"},
    "MR3I":  {"db": 31, "offset": 144,  "type": "float"},
    "MR3F":  {"db": 31, "offset": 148,  "type": "float"},
    "MR4I":  {"db": 31, "offset": 152,  "type": "float"},
    "MR4F":  {"db": 31, "offset": 156,  "type": "float"},
    "MR5I":  {"db": 31, "offset": 160,  "type": "float"},
    "MR5F":  {"db": 31, "offset": 164,  "type": "float"},
    "MR6I":  {"db": 31, "offset": 168,  "type": "float"},
    "MR6F":  {"db": 31, "offset": 172,  "type": "float"},
    "MR7I":  {"db": 31, "offset": 176,  "type": "float"},
    "MR7F":  {"db": 31, "offset": 180,  "type": "float"},
    "MR8I":  {"db": 31, "offset": 184,  "type": "float"},
    "MR8F":  {"db": 31, "offset": 188,  "type": "float"},
    "MR9I":  {"db": 31, "offset": 192,  "type": "float"},
    "MR9F":  {"db": 31, "offset": 196,  "type": "float"},
    "MR10I": {"db": 31, "offset": 200,  "type": "float"},
    "MR10F": {"db": 31, "offset": 204,  "type": "float"},
}

PATTERN_DB_MAP = {
    "MP1I":  {"db": 31, "offset": 208,  "type": "float"},
    "MP1F":  {"db": 31, "offset": 212,  "type": "float"},
    "MP2I":  {"db": 31, "offset": 216,  "type": "float"},
    "MP2F":  {"db": 31, "offset": 220,  "type": "float"},
    "MP3I":  {"db": 31, "offset": 224,  "type": "float"},
    "MP3F":  {"db": 31, "offset": 228,  "type": "float"},
    "MP4I":  {"db": 31, "offset": 232,  "type": "float"},
    "MP4F":  {"db": 31, "offset": 236,  "type": "float"},
    "MP5I":  {"db": 31, "offset": 240,  "type": "float"},
    "MP5F":  {"db": 31, "offset": 244,  "type": "float"},
    "MP6I":  {"db": 31, "offset": 248,  "type": "float"},
    "MP6F":  {"db": 31, "offset": 252,  "type": "float"},
    "MP7I":  {"db": 31, "offset": 256,  "type": "float"},
    "MP7F":  {"db": 31, "offset": 260,  "type": "float"},
    "MP8I":  {"db": 31, "offset": 264,  "type": "float"},
    "MP8F":  {"db": 31, "offset": 268,  "type": "float"},
    "MP9I":  {"db": 31, "offset": 272,  "type": "float"},
    "MP9F":  {"db": 31, "offset": 276,  "type": "float"},
    "MP10I": {"db": 31, "offset": 280,  "type": "float"},
    "MP10F": {"db": 31, "offset": 284,  "type": "float"},
}

# ------------------------------------------------------------------------------
#                   VARIABLES / OBJETOS GLOBALES
# ------------------------------------------------------------------------------
cliente_plc = snap7.client.Client()
conectado_plc = False

ultimo_dato_calibracion = None
lock = threading.Lock()

# ------------------------------------------------------------------------------
#                   FUNCIONES PARA CSV Y LOG
# ------------------------------------------------------------------------------
def guardar_evento(evento, tipo, valor, valor_inicial, diferencia="", setpoint=""):
    """
    Guarda un evento en el CSV con la información dada.
    """
    if tipo == "Patron":
        try:
            valor = round(float(valor), 3)
            valor_inicial = round(float(valor_inicial), 3)
            if diferencia != "":
                diferencia = round(float(diferencia), 3)
        except Exception as e:
            logging.error(f"Error redondeando valores para Patron: {e}")

    file_exists = os.path.isfile(CSV_FILE)
    write_header = not file_exists or os.stat(CSV_FILE).st_size == 0

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        writer.writerow([evento, tipo, valor, valor_inicial, diferencia, setpoint, timestamp])

# ------------------------------------------------------------------------------
#                   NUEVA FUNCIÓN: Escribir bandera en DB250 para repeticiones
# ------------------------------------------------------------------------------

# --- NUEVO --- Configuración de la DB de banderas para repeticiones (DB250)
# Se utilizarán los bits: 1 para rep1, 2 para rep2, …, 10 para rep10.
BANDERAS_DB = 3
BANDERAS_BYTE73 = 73  # Se usa el byte 0

BANDERAS_BYTE74 = 74

def escribir_bandera_db73(rep: int, valor: bool):
    try:
        # Leer el byte actual del PLC
        data = cliente_plc.db_read(BANDERAS_DB, BANDERAS_BYTE73, 1)
        # Actualizar únicamente el bit deseado
        set_bool(data, 0, rep, valor)
        # Escribir el byte modificado de vuelta al PLC
        cliente_plc.db_write(BANDERAS_DB, BANDERAS_BYTE73, data)
        logging.info(f"Escrito bandera en DB{BANDERAS_DB}.DBX{BANDERAS_BYTE74}.{rep} a {valor}")
    except Exception as e:
        logging.error(f"Error escribiendo bandera en DB{BANDERAS_DB}.DBX{BANDERAS_BYTE74}.{rep}: {e}")


def escribir_bandera_db74(rep: int, valor: bool):
    try:
        # Leer el byte actual del PLC
        data = cliente_plc.db_read(BANDERAS_DB, BANDERAS_BYTE74, 1)
        # Actualizar únicamente el bit deseado
        set_bool(data, 0, rep, valor)
        # Escribir el byte modificado de vuelta al PLC
        cliente_plc.db_write(BANDERAS_DB, BANDERAS_BYTE74, data)
        logging.info(f"Escrito bandera en DB{BANDERAS_DB}.DBX{BANDERAS_BYTE74}.{rep} a {valor}")
    except Exception as e:
        logging.error(f"Error escribiendo bandera en DB{BANDERAS_DB}.DBX{BANDERAS_BYTE74}.{rep}: {e}")


# ------------------------------------------------------------------------------
#                   FUNCIONES PARA CONEXIÓN Y LECTURA DE PLC
# ------------------------------------------------------------------------------
def conectar_plc():
    global conectado_plc
    try:
        cliente_plc.connect(PLC_IP, PLC_RACK, PLC_SLOT)
        if cliente_plc.get_connected():
            logging.info(f"Conectado al PLC en {PLC_IP}, rack={PLC_RACK}, slot={PLC_SLOT}")
            conectado_plc = True
    except Exception as e:
        logging.error(f"Error conectando al PLC: {e}")
        conectado_plc = False

def leer_bandera_test():
    if not conectado_plc:
        return False
    try:
        data = cliente_plc.db_read(FLAG_DB, FLAG_OFFSET, 3)
        return get_bool(data, 0, 3)
    except Exception as e:
        logging.error(f"Error leyendo bandera test: {e}")
        return False

def escribir_bandera_test(valor: bool):
    if not conectado_plc:
        return
    try:
        data = bytearray(1)
        set_bool(data, 0, 0, valor)
        cliente_plc.db_write(FLAG_DB, FLAG_OFFSET, data)
    except Exception as e:
        logging.error(f"Error escribiendo bandera test: {e}")

def leer_dbd_real(db_num, start):
    """
    Lee 4 bytes e interpreta el valor como REAL (IEEE754 big-endian).
    """
    if not conectado_plc:
        return 0.0
    try:
        data = cliente_plc.db_read(db_num, start, 4)
        return struct.unpack('>f', data)[0]
    except Exception as e:
        logging.error(f"Error leyendo DB{db_num}.DBD{start} como REAL: {e}")
        return 0.0

def leer_dbd_int(db_num, start):
    """
    Lee 4 bytes e interpreta el valor como entero (32-bit signed, big-endian).
    """
    if not conectado_plc:
        return 0
    try:
        data = cliente_plc.db_read(db_num, start, 2)
        return struct.unpack('>h', data)[0]
    except Exception as e:
        logging.error(f"Error leyendo DB{db_num}.DBD{start} como entero: {e}")
        return 0

# ------------------------------------------------------------------------------
#   FUNCIÓN ESCRITURA GENÉRICA: escribe valor según tag (MRx, MPx) en DB
# ------------------------------------------------------------------------------
def escribir_valor_plc(tag, valor, cliente_plc):
    """
    Dependiendo del tag (p.ej. "MR2I", "MP1F"), se busca en los diccionarios
    CAMERA_DB_MAP o PATTERN_DB_MAP la configuración {db, offset, type} y se hace la escritura.
    """
    config = None
    if tag in CAMERA_DB_MAP:
        config = CAMERA_DB_MAP[tag]
    elif tag in PATTERN_DB_MAP:
        config = PATTERN_DB_MAP[tag]
    else:
        raise ValueError(f"Tag '{tag}' no encontrado en la configuración de DB.")

    db = config["db"]
    offset = config["offset"]
    tipo = config["type"]

    try:
        if tipo == "int":
            data = bytearray(struct.pack('>i', int(valor)))
        elif tipo == "float":
            data = bytearray(struct.pack('>f', float(valor)))
        else:
            raise ValueError(f"Tipo de dato '{tipo}' no soportado.")
        cliente_plc.db_write(db, offset, data)
        logging.info(f"Escrito valor={valor} en DB{db}.DBD{offset} (tag={tag}, tipo={tipo})")
    except Exception as e:
        logging.error(f"Error escribiendo valor en PLC para tag={tag}, valor={valor}: {e}")

# ------------------------------------------------------------------------------
#                   ENDPOINT FLASK PARA RECIBIR DATOS DE CÁMARA
# ------------------------------------------------------------------------------
@app.route('/calibration', methods=['POST'])
def calibration():
    global ultimo_dato_calibracion
    data = request.get_json()
    if not data or 'dato' not in data:
        return jsonify({'error': 'No se proporcionó el dato'}), 400
    with lock:
        ultimo_dato_calibracion = data['dato']
    logging.info("Valor de cámara actualizado: %s", data['dato'])
    return jsonify({'status': 'Dato actualizado', 'dato': data['dato']}), 200

def obtener_num_repeticiones():
    try:
        # Crear una nueva instancia de cliente para esta lectura
        cliente_temp = snap7.client.Client()
        cliente_temp.connect(PLC_IP, PLC_RACK, PLC_SLOT)
        # Leer 2 bytes desde DB3 en offset 26 (DB3.DBW26)
        data = cliente_temp.db_read(3, 26, 2)
        # Desempaquetar como entero de 16 bits (big-endian)
        num = struct.unpack('>h', data)[0]
        cliente_temp.disconnect()
        logging.info(f"NUM_REPETICIONES obtenido desde DB3.DBW26: {num}")
        return num
    except Exception as e:
        logging.error(f"Error leyendo NUM_REPETICIONES desde DB3.DBW26: {e}")
        return 10  # Valor por defecto en caso de error
    





# ------------------------------------------------------------------------------
#                   HILO PRINCIPAL DE PROCESAMIENTO (CALIBRACIÓN)
# ------------------------------------------------------------------------------
def procesamiento_dato():
    global ultimo_dato_calibracion
    while True:
        if not conectado_plc:
            conectar_plc()

        bandera = leer_bandera_test()
        if bandera:
            NUM_REPETICIONES = obtener_num_repeticiones()
            print('NUMERO DE REPETICIONES', obtener_num_repeticiones())
            logging.info("La bandera está en TRUE, comenzamos la lógica de calibración para %s.", TEST)
            
            # Esperar a que llegue un valor nuevo para iniciar
            with lock:
                primer_valor = ultimo_dato_calibracion
            while True:
                with lock:
                    nuevo_valor = ultimo_dato_calibracion
                if nuevo_valor != primer_valor:
                    valor_inicial_camara = nuevo_valor
                    break
                time.sleep(0.05)
            logging.info("Inicia calibración con valor inicial de cámara: %s", valor_inicial_camara)

            # Leer patrón y setpoint
            v_patron_inicial = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
            setpoint = leer_dbd_int(SETPOINT_DB, SETPOINT_OFFSET)
            logging.info("Patrón inicial=%s, Setpoint=%s", v_patron_inicial, setpoint)

            # Repetición 1: se asume que la bandera (DB250.DBX0.1) ya viene en TRUE desde el PLC.
            guardar_evento("MR1I", "Camara", valor_inicial_camara, valor_inicial_camara, "", setpoint)
            guardar_evento("MP1I", "Patron", v_patron_inicial, v_patron_inicial, "", setpoint)
            
            escribir_bandera_db73(3, True)


            escribir_valor_plc("MR1I", valor_inicial_camara, cliente_plc)
            escribir_valor_plc("MP1I", v_patron_inicial, cliente_plc)
            
            # Generar thresholds para las repeticiones
            thresholds = []
            thresholds.append(valor_inicial_camara + setpoint)  # Repetición 1
            for i in range(1, 10):  # Repeticiones 2 a 10
                thresholds.append(thresholds[i-1] + GAP)

            # Arrays para almacenar valores de inicio y final de cámara y patrón
            r_inicial_camara = [None] * 10
            r_inicial_patron = [None] * 10
            r_final_camara = [None] * 10
            r_final_patron = [None] * 10

            rep_inicio_registrado = [False] * 10
            rep_final_registrado = [False] * 10

            # Repetición 1: ya iniciada; su bandera permanece TRUE hasta que se finalice.
            r_inicial_camara[0] = valor_inicial_camara
            r_inicial_patron[0] = v_patron_inicial
            rep_inicio_registrado[0] = True

            while True:
                with lock:
                    v_camara = ultimo_dato_calibracion
                if v_camara is None:
                    time.sleep(0.1)
                    continue

                # --- Repetición 1: Finalización ---
                if NUM_REPETICIONES >= 1:
                    if (not rep_final_registrado[0]) and (v_camara >= thresholds[0]):
                        r_final_camara[0] = v_camara
                        r_final_patron[0] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                        dif_cam = r_final_camara[0] - r_inicial_camara[0]
                        dif_pat = r_final_patron[0] - r_inicial_patron[0]

                        guardar_evento("MR1F", "Camara", r_final_camara[0], r_inicial_camara[0], dif_cam)
                        guardar_evento("MP1F", "Patron", r_final_patron[0], r_inicial_patron[0], dif_pat)
                        escribir_valor_plc("MR1F", r_final_camara[0], cliente_plc)
                        escribir_valor_plc("MP1F", r_final_patron[0], cliente_plc)
                        escribir_bandera_db73(3, False) 
                        rep_final_registrado[0] = True
                         # Solo se envía FALSE al finalizar rep1
                        logging.info("Repetición 1 final: MR1F=%.2f, MP1F=%.2f | difCam=%.2f, difPat=%.2f",
                                     r_final_camara[0], r_final_patron[0], dif_cam, dif_pat)

                # --- Repetición 2: Inicio y Final ---
                if NUM_REPETICIONES >= 2:
                    if (r_inicial_camara[0] is not None) and (not rep_inicio_registrado[1]):
                        if v_camara >= (r_inicial_camara[0] + GAP):
                            r_inicial_camara[1] = v_camara
                            r_inicial_patron[1] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                            guardar_evento("MR2I", "Camara", r_inicial_camara[1], r_inicial_camara[1])
                            guardar_evento("MP2I", "Patron", r_inicial_patron[1], r_inicial_patron[1])
                            escribir_valor_plc("MR2I", r_inicial_camara[1], cliente_plc)
                            escribir_valor_plc("MP2I", r_inicial_patron[1], cliente_plc)
                            rep_inicio_registrado[1] = True
                            escribir_bandera_db73(4, True)
                            logging.info("Repetición 2 inicio: MR2I=%.2f, MP2I=%.2f",
                                         r_inicial_camara[1], r_inicial_patron[1])
                    if rep_inicio_registrado[1] and (not rep_final_registrado[1]) and (v_camara >= thresholds[1]):
                        r_final_camara[1] = v_camara
                        r_final_patron[1] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                        dif_cam = r_final_camara[1] - r_inicial_camara[1]
                        dif_pat = r_final_patron[1] - r_inicial_patron[1]
                        guardar_evento("MR2F", "Camara", r_final_camara[1], r_inicial_camara[1], dif_cam)
                        guardar_evento("MP2F", "Patron", r_final_patron[1], r_inicial_patron[1], dif_pat)
                        escribir_valor_plc("MR2F", r_final_camara[1], cliente_plc)
                        escribir_valor_plc("MP2F", r_final_patron[1], cliente_plc)
                        rep_final_registrado[1] = True
                        escribir_bandera_db73(4, False)  # Enviar FALSE solo al finalizar rep2
                        logging.info("Repetición 2 final: MR2F=%.2f, MP2F=%.2f | difCam=%.2f, difPat=%.2f",
                                     r_final_camara[1], r_final_patron[1], dif_cam, dif_pat)

                # --- Repetición 3: Inicio y Final ---
                if NUM_REPETICIONES >= 3:
                    if (r_inicial_camara[1] is not None) and (not rep_inicio_registrado[2]):
                        if v_camara >= (r_inicial_camara[1] + GAP):
                            r_inicial_camara[2] = v_camara
                            r_inicial_patron[2] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                            guardar_evento("MR3I", "Camara", r_inicial_camara[2], r_inicial_camara[2])
                            guardar_evento("MP3I", "Patron", r_inicial_patron[2], r_inicial_patron[2])
                            escribir_valor_plc("MR3I", r_inicial_camara[2], cliente_plc)
                            escribir_valor_plc("MP3I", r_inicial_patron[2], cliente_plc)
                            rep_inicio_registrado[2] = True
                            escribir_bandera_db73(5, True)
                            logging.info("Repetición 3 inicio: MR3I=%.2f, MP3I=%.2f",
                                         r_inicial_camara[2], r_inicial_patron[2])
                    if rep_inicio_registrado[2] and (not rep_final_registrado[2]) and (v_camara >= thresholds[2]):
                        r_final_camara[2] = v_camara
                        r_final_patron[2] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                        dif_cam = r_final_camara[2] - r_inicial_camara[2]
                        dif_pat = r_final_patron[2] - r_inicial_patron[2]
                        guardar_evento("MR3F", "Camara", r_final_camara[2], r_inicial_camara[2], dif_cam)
                        guardar_evento("MP3F", "Patron", r_final_patron[2], r_inicial_patron[2], dif_pat)
                        escribir_valor_plc("MR3F", r_final_camara[2], cliente_plc)
                        escribir_valor_plc("MP3F", r_final_patron[2], cliente_plc)
                        rep_final_registrado[2] = True
                        escribir_bandera_db73(5, False)
                        logging.info("Repetición 3 final: MR3F=%.2f, MP3F=%.2f | difCam=%.2f, difPat=%.2f",
                                     r_final_camara[2], r_final_patron[2], dif_cam, dif_pat)

                # --- Repetición 4: Inicio y Final ---
                if NUM_REPETICIONES >= 4:
                    if (r_inicial_camara[2] is not None) and (not rep_inicio_registrado[3]):
                        if v_camara >= (r_inicial_camara[2] + GAP):
                            r_inicial_camara[3] = v_camara
                            r_inicial_patron[3] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                            guardar_evento("MR4I", "Camara", r_inicial_camara[3], r_inicial_camara[3])
                            guardar_evento("MP4I", "Patron", r_inicial_patron[3], r_inicial_patron[3])
                            escribir_valor_plc("MR4I", r_inicial_camara[3], cliente_plc)
                            escribir_valor_plc("MP4I", r_inicial_patron[3], cliente_plc)
                            rep_inicio_registrado[3] = True
                            escribir_bandera_db73(6, True)
                            logging.info("Repetición 4 inicio: MR4I=%.2f, MP4I=%.2f",
                                         r_inicial_camara[3], r_inicial_patron[3])
                    if rep_inicio_registrado[3] and (not rep_final_registrado[3]) and (v_camara >= thresholds[3]):
                        r_final_camara[3] = v_camara
                        r_final_patron[3] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                        dif_cam = r_final_camara[3] - r_inicial_camara[3]
                        dif_pat = r_final_patron[3] - r_inicial_patron[3]
                        guardar_evento("MR4F", "Camara", r_final_camara[3], r_inicial_camara[3], dif_cam)
                        guardar_evento("MP4F", "Patron", r_final_patron[3], r_inicial_patron[3], dif_pat)
                        escribir_valor_plc("MR4F", r_final_camara[3], cliente_plc)
                        escribir_valor_plc("MP4F", r_final_patron[3], cliente_plc)
                        rep_final_registrado[3] = True
                        escribir_bandera_db73(6, False)
                        logging.info("Repetición 4 final: MR4F=%.2f, MP4F=%.2f | difCam=%.2f, difPat=%.2f",
                                     r_final_camara[3], r_final_patron[3], dif_cam, dif_pat)

                # --- Repetición 5: Inicio y Final ---
                if NUM_REPETICIONES >= 5:
                    if (r_inicial_camara[3] is not None) and (not rep_inicio_registrado[4]):
                        if v_camara >= (r_inicial_camara[3] + GAP):
                            r_inicial_camara[4] = v_camara
                            r_inicial_patron[4] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                            guardar_evento("MR5I", "Camara", r_inicial_camara[4], r_inicial_camara[4])
                            guardar_evento("MP5I", "Patron", r_inicial_patron[4], r_inicial_patron[4])
                            escribir_valor_plc("MR5I", r_inicial_camara[4], cliente_plc)
                            escribir_valor_plc("MP5I", r_inicial_patron[4], cliente_plc)
                            rep_inicio_registrado[4] = True
                            escribir_bandera_db73(7, True)
                            logging.info("Repetición 5 inicio: MR5I=%.2f, MP5I=%.2f",
                                         r_inicial_camara[4], r_inicial_patron[4])
                    if rep_inicio_registrado[4] and (not rep_final_registrado[4]) and (v_camara >= thresholds[4]):
                        r_final_camara[4] = v_camara
                        r_final_patron[4] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                        dif_cam = r_final_camara[4] - r_inicial_camara[4]
                        dif_pat = r_final_patron[4] - r_inicial_patron[4]
                        guardar_evento("MR5F", "Camara", r_final_camara[4], r_inicial_camara[4], dif_cam)
                        guardar_evento("MP5F", "Patron", r_final_patron[4], r_inicial_patron[4], dif_pat)
                        escribir_valor_plc("MR5F", r_final_camara[4], cliente_plc)
                        escribir_valor_plc("MP5F", r_final_patron[4], cliente_plc)
                        rep_final_registrado[4] = True
                        escribir_bandera_db73(7, False)
                        logging.info("Repetición 5 final: MR5F=%.2f, MP5F=%.2f | difCam=%.2f, difPat=%.2f",
                                     r_final_camara[4], r_final_patron[4], dif_cam, dif_pat)

                # --- Repetición 6: Inicio y Final ---
                if NUM_REPETICIONES >= 6:
                    if (r_inicial_camara[4] is not None) and (not rep_inicio_registrado[5]):
                        if v_camara >= (r_inicial_camara[4] + GAP):
                            r_inicial_camara[5] = v_camara
                            r_inicial_patron[5] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                            guardar_evento("MR6I", "Camara", r_inicial_camara[5], r_inicial_camara[5])
                            guardar_evento("MP6I", "Patron", r_inicial_patron[5], r_inicial_patron[5])
                            escribir_valor_plc("MR6I", r_inicial_camara[5], cliente_plc)
                            escribir_valor_plc("MP6I", r_inicial_patron[5], cliente_plc)
                            rep_inicio_registrado[5] = True
                            escribir_bandera_db74(0, True)
                            logging.info("Repetición 6 inicio: MR6I=%.2f, MP6I=%.2f",
                                         r_inicial_camara[5], r_inicial_patron[5])
                    if rep_inicio_registrado[5] and (not rep_final_registrado[5]) and (v_camara >= thresholds[5]):
                        r_final_camara[5] = v_camara
                        r_final_patron[5] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                        dif_cam = r_final_camara[5] - r_inicial_camara[5]
                        dif_pat = r_final_patron[5] - r_inicial_patron[5]
                        guardar_evento("MR6F", "Camara", r_final_camara[5], r_inicial_camara[5], dif_cam)
                        guardar_evento("MP6F", "Patron", r_final_patron[5], r_inicial_patron[5], dif_pat)
                        escribir_valor_plc("MR6F", r_final_camara[5], cliente_plc)
                        escribir_valor_plc("MP6F", r_final_patron[5], cliente_plc)
                        rep_final_registrado[5] = True
                        escribir_bandera_db74(0, False)
                        logging.info("Repetición 6 final: MR6F=%.2f, MP6F=%.2f | difCam=%.2f, difPat=%.2f",
                                     r_final_camara[5], r_final_patron[5], dif_cam, dif_pat)

                # --- Repetición 7: Inicio y Final ---
                if NUM_REPETICIONES >= 7:
                    if (r_inicial_camara[5] is not None) and (not rep_inicio_registrado[6]):
                        if v_camara >= (r_inicial_camara[5] + GAP):
                            r_inicial_camara[6] = v_camara
                            r_inicial_patron[6] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                            guardar_evento("MR7I", "Camara", r_inicial_camara[6], r_inicial_camara[6])
                            guardar_evento("MP7I", "Patron", r_inicial_patron[6], r_inicial_patron[6])
                            escribir_valor_plc("MR7I", r_inicial_camara[6], cliente_plc)
                            escribir_valor_plc("MP7I", r_inicial_patron[6], cliente_plc)
                            rep_inicio_registrado[6] = True
                            escribir_bandera_db74(1, True)
                            logging.info("Repetición 7 inicio: MR7I=%.2f, MP7I=%.2f",
                                         r_inicial_camara[6], r_inicial_patron[6])
                    if rep_inicio_registrado[6] and (not rep_final_registrado[6]) and (v_camara >= thresholds[6]):
                        r_final_camara[6] = v_camara
                        r_final_patron[6] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                        dif_cam = r_final_camara[6] - r_inicial_camara[6]
                        dif_pat = r_final_patron[6] - r_inicial_patron[6]
                        guardar_evento("MR7F", "Camara", r_final_camara[6], r_inicial_camara[6], dif_cam)
                        guardar_evento("MP7F", "Patron", r_final_patron[6], r_inicial_patron[6], dif_pat)
                        escribir_valor_plc("MR7F", r_final_camara[6], cliente_plc)
                        escribir_valor_plc("MP7F", r_final_patron[6], cliente_plc)
                        rep_final_registrado[6] = True
                        escribir_bandera_db74(1, False)
                        logging.info("Repetición 7 final: MR7F=%.2f, MP7F=%.2f | difCam=%.2f, difPat=%.2f",
                                     r_final_camara[6], r_final_patron[6], dif_cam, dif_pat)

                # --- Repetición 8: Inicio y Final ---
                if NUM_REPETICIONES >= 8:
                    if (r_inicial_camara[6] is not None) and (not rep_inicio_registrado[7]):
                        if v_camara >= (r_inicial_camara[6] + GAP):
                            r_inicial_camara[7] = v_camara
                            r_inicial_patron[7] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                            guardar_evento("MR8I", "Camara", r_inicial_camara[7], r_inicial_camara[7])
                            guardar_evento("MP8I", "Patron", r_inicial_patron[7], r_inicial_patron[7])
                            escribir_valor_plc("MR8I", r_inicial_camara[7], cliente_plc)
                            escribir_valor_plc("MP8I", r_inicial_patron[7], cliente_plc)
                            rep_inicio_registrado[7] = True
                            escribir_bandera_db74(2, True)
                            logging.info("Repetición 8 inicio: MR8I=%.2f, MP8I=%.2f",
                                         r_inicial_camara[7], r_inicial_patron[7])
                    if rep_inicio_registrado[7] and (not rep_final_registrado[7]) and (v_camara >= thresholds[7]):
                        r_final_camara[7] = v_camara
                        r_final_patron[7] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                        dif_cam = r_final_camara[7] - r_inicial_camara[7]
                        dif_pat = r_final_patron[7] - r_inicial_patron[7]
                        guardar_evento("MR8F", "Camara", r_final_camara[7], r_inicial_camara[7], dif_cam)
                        guardar_evento("MP8F", "Patron", r_final_patron[7], r_inicial_patron[7], dif_pat)
                        escribir_valor_plc("MR8F", r_final_camara[7], cliente_plc)
                        escribir_valor_plc("MP8F", r_final_patron[7], cliente_plc)
                        rep_final_registrado[7] = True
                        escribir_bandera_db74(2, False)
                        logging.info("Repetición 8 final: MR8F=%.2f, MP8F=%.2f | difCam=%.2f, difPat=%.2f",
                                     r_final_camara[7], r_final_patron[7], dif_cam, dif_pat)

                # --- Repetición 9: Inicio y Final ---
                if NUM_REPETICIONES >= 9:
                    if (r_inicial_camara[7] is not None) and (not rep_inicio_registrado[8]):
                        if v_camara >= (r_inicial_camara[7] + GAP):
                            r_inicial_camara[8] = v_camara
                            r_inicial_patron[8] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                            guardar_evento("MR9I", "Camara", r_inicial_camara[8], r_inicial_camara[8])
                            guardar_evento("MP9I", "Patron", r_inicial_patron[8], r_inicial_patron[8])
                            escribir_valor_plc("MR9I", r_inicial_camara[8], cliente_plc)
                            escribir_valor_plc("MP9I", r_inicial_patron[8], cliente_plc)
                            rep_inicio_registrado[8] = True
                            escribir_bandera_db74(3, True)
                            logging.info("Repetición 9 inicio: MR9I=%.2f, MP9I=%.2f",
                                         r_inicial_camara[8], r_inicial_patron[8])
                    if rep_inicio_registrado[8] and (not rep_final_registrado[8]) and (v_camara >= thresholds[8]):
                        r_final_camara[8] = v_camara
                        r_final_patron[8] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                        dif_cam = r_final_camara[8] - r_inicial_camara[8]
                        dif_pat = r_final_patron[8] - r_inicial_patron[8]
                        guardar_evento("MR9F", "Camara", r_final_camara[8], r_inicial_camara[8], dif_cam)
                        guardar_evento("MP9F", "Patron", r_final_patron[8], r_inicial_patron[8], dif_pat)
                        escribir_valor_plc("MR9F", r_final_camara[8], cliente_plc)
                        escribir_valor_plc("MP9F", r_final_patron[8], cliente_plc)
                        rep_final_registrado[8] = True
                        escribir_bandera_db74(3, False)
                        logging.info("Repetición 9 final: MR9F=%.2f, MP9F=%.2f | difCam=%.2f, difPat=%.2f",
                                     r_final_camara[8], r_final_patron[8], dif_cam, dif_pat)

                # --- Repetición 10: Inicio y Final ---
                if NUM_REPETICIONES >= 10:
                    if (r_inicial_camara[8] is not None) and (not rep_inicio_registrado[9]):
                        if v_camara >= (r_inicial_camara[8] + GAP):
                            r_inicial_camara[9] = v_camara
                            r_inicial_patron[9] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                            guardar_evento("MR10I", "Camara", r_inicial_camara[9], r_inicial_camara[9])
                            guardar_evento("MP10I", "Patron", r_inicial_patron[9], r_inicial_patron[9])
                            escribir_valor_plc("MR10I", r_inicial_camara[9], cliente_plc)
                            escribir_valor_plc("MP10I", r_inicial_patron[9], cliente_plc)
                            rep_inicio_registrado[9] = True
                            escribir_bandera_db74(4, True)
                            logging.info("Repetición 10 inicio: MR10I=%.2f, MP10I=%.2f",
                                         r_inicial_camara[9], r_inicial_patron[9])
                    if rep_inicio_registrado[9] and (not rep_final_registrado[9]) and (v_camara >= thresholds[9]):
                        r_final_camara[9] = v_camara
                        r_final_patron[9] = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
                        dif_cam = r_final_camara[9] - r_inicial_camara[9]
                        dif_pat = r_final_patron[9] - r_inicial_patron[9]
                        guardar_evento("MR10F", "Camara", r_final_camara[9], r_inicial_camara[9], dif_cam)
                        guardar_evento("MP10F", "Patron", r_final_patron[9], r_inicial_patron[9], dif_pat)
                        escribir_valor_plc("MR10F", r_final_camara[9], cliente_plc)
                        escribir_valor_plc("MP10F", r_final_patron[9], cliente_plc)
                        rep_final_registrado[9] = True
                        escribir_bandera_db74(4, False)
                        logging.info("Repetición 10 final: MR10F=%.2f, MP10F=%.2f | difCam=%.2f, difPat=%.2f",
                                     r_final_camara[9], r_final_patron[9], dif_cam, dif_pat)

                # Si todas las repeticiones solicitadas finalizaron, se envía FALSE en la bandera general
                if all(rep_final_registrado[:NUM_REPETICIONES]):
                    escribir_bandera_test(False)
                    logging.info("Fin de calibración. Se pone la bandera general en FALSE.")
                    break

                time.sleep(0.1)
        else:
            time.sleep(0.3)

# ------------------------------------------------------------------------------
#                   FUNCIONES DE ARRANQUE Y CIERRE
# ------------------------------------------------------------------------------
def iniciar_hilos():
    conectar_plc()
    if conectado_plc:
        bandera_inicial = leer_bandera_test()
        patron_inicial = leer_dbd_real(PATTERN_DB, PATTERN_OFFSET)
        setpoint_inicial = leer_dbd_int(SETPOINT_DB, SETPOINT_OFFSET)
        logging.info("Al iniciar: bandera=%s, patrón=%s, setpoint=%s",
                     bandera_inicial, patron_inicial, setpoint_inicial)

    hilo_calibracion = threading.Thread(target=procesamiento_dato, daemon=True)
    hilo_calibracion.start()

def cerrar_conexion_plc():
    if conectado_plc:
        cliente_plc.disconnect()
        logging.info("Conexión al PLC cerrada.")

atexit.register(cerrar_conexion_plc)

# ------------------------------------------------------------------------------
#                   MAIN
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    iniciar_hilos()
    app.run(host='0.0.0.0', port=5001, threaded=True)
