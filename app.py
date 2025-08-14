from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import mediapipe as mp
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
try:
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError:
    rdMolDraw2D = None
import math
import json
import io
import base64
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========== Configuración MediaPipe ==========
mp_hands = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ========== Sistema de Moléculas Global ==========
class MolecularSystem:
    def __init__(self):
        self.current_molecule = None
        self.molecule_data = {}
        self.molecules = {
            'primary': {
                'pos': [300, 250],
                'scale': 1.0,
                'rotation_x': 0,
                'rotation_y': 0,
                'rotation_z': 0,
                'active': False,
                'smiles': None,
                'name': None
            }
        }
        self.interaction_distance = 150
        self.binding_strength = 0.0
        self.is_bound = False
        
    def load_molecule_from_smiles(self, smiles, name):
        """Carga una molécula desde código SMILES"""
        try:
            # Validar SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'success': False, 'error': 'SMILES inválido'}
            
            # Generar conformación 3D
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
            
            # Calcular propiedades
            properties = {
                'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
                'logp': rdMolDescriptors.Crippen.MolLogP(mol),
                'hbd': rdMolDescriptors.CalcNumHBD(mol),
                'hba': rdMolDescriptors.CalcNumHBA(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol)
            }
            
            # Generar imagen de la molécula
            img = Draw.MolToImage(mol, size=(400, 400))
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_rgba = cv2.cvtColor(img_cv, cv2.COLOR_BGR2BGRA)
            
            # Hacer fondo transparente
            white_pixels = np.all(img_rgba[:, :, :3] > [240, 240, 240], axis=2)
            img_rgba[white_pixels] = [0, 0, 0, 0]
            
            # Guardar datos
            self.current_molecule = {
                'smiles': smiles,
                'name': name,
                'mol': mol,
                'properties': properties,
                'image': img_rgba
            }
            
            # Actualizar molécula principal
            self.molecules['primary']['smiles'] = smiles
            self.molecules['primary']['name'] = name
            self.molecules['primary']['active'] = True
            
            return {
                'success': True, 
                'message': f'Molécula {name} cargada exitosamente',
                'properties': properties
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def validate_smiles(self, smiles):
        """Valida un código SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'valid': False, 'error': 'Estructura química inválida'}
            
            # Verificaciones adicionales
            num_atoms = mol.GetNumAtoms()
            if num_atoms == 0:
                return {'valid': False, 'error': 'La molécula no contiene átomos'}
            
            if num_atoms > 200:
                return {'valid': False, 'error': 'Molécula demasiado grande (>200 átomos)'}
            
            return {
                'valid': True, 
                'message': 'SMILES válido',
                'num_atoms': num_atoms
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Error de validación: {str(e)}'}
    
    def update_molecule_position(self, hand_landmarks, frame_shape):
        """Actualiza posición de molécula basada en gestos"""
        if not self.current_molecule:
            return
        
        h, w, _ = frame_shape
        
        # Obtener puntos clave de la mano
        palm_center = hand_landmarks.landmark[9]
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Convertir a coordenadas de pantalla
        palm_pos = (int(palm_center.x * w), int(palm_center.y * h))
        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        
        # Detectar gestos
        thumb_index_dist = math.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
        
        # Gesto de agarre (mover)
        if thumb_index_dist < 50:
            self.molecules['primary']['pos'] = [palm_pos[0] - 100, palm_pos[1] - 100]
            self.molecules['primary']['active'] = True
        
        # Gesto de zoom
        elif thumb_index_dist > 80:
            scale = min(max(thumb_index_dist / 100, 0.5), 2.5)
            self.molecules['primary']['scale'] = scale
            self.molecules['primary']['active'] = True
        
        # Auto-rotación suave
        else:
            self.molecules['primary']['active'] = False
            self.molecules['primary']['rotation_y'] += 0.01
    
    def get_molecular_data(self):
        """Obtiene datos actuales del sistema molecular"""
        return {
            'current_molecule': {
                'name': self.current_molecule['name'] if self.current_molecule else None,
                'smiles': self.current_molecule['smiles'] if self.current_molecule else None,
                'properties': self.current_molecule['properties'] if self.current_molecule else None
            },
            'molecules': self.molecules,
            'interaction_status': {
                'is_bound': self.is_bound,
                'binding_strength': self.binding_strength
            }
        }
    
    def reset(self):
        """Reinicia el sistema molecular"""
        self.current_molecule = None
        self.molecules['primary'] = {
            'pos': [300, 250],
            'scale': 1.0,
            'rotation_x': 0,
            'rotation_y': 0,
            'rotation_z': 0,
            'active': False,
            'smiles': None,
            'name': None
        }
        self.binding_strength = 0.0
        self.is_bound = False

# Inicializar sistema molecular
molecular_system = MolecularSystem()

def dibujar_overlay_molecular(frame, overlay, pos, scale=1.0, alpha=1.0):
    """Dibuja overlay molecular con efectos"""
    try:
        if overlay is None:
            return
        
        if scale != 1.0:
            h, w = overlay.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            overlay = cv2.resize(overlay, (new_w, new_h))
        
        x, y = pos
        y1, y2 = max(0, y), min(frame.shape[0], y + overlay.shape[0])
        x1, x2 = max(0, x), min(frame.shape[1], x + overlay.shape[1])
        
        if y2 > y1 and x2 > x1:
            oy1 = max(0, -y)
            oy2 = overlay.shape[0] - max(0, y + overlay.shape[0] - frame.shape[0])
            ox1 = max(0, -x)
            ox2 = overlay.shape[1] - max(0, x + overlay.shape[1] - frame.shape[1])
            
            roi = frame[y1:y2, x1:x2]
            overlay_region = overlay[oy1:oy2, ox1:ox2]
            
            if overlay_region.shape[0] > 0 and overlay_region.shape[1] > 0:
                mask = (overlay_region[:, :, 3] * alpha).astype(np.uint8)
                mask_inv = cv2.bitwise_not(mask)
                
                img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                img_fg = cv2.bitwise_and(overlay_region[:, :, :3], overlay_region[:, :, :3], mask=mask)
                dst = cv2.add(img_bg, img_fg)
                frame[y1:y2, x1:x2] = dst
                
    except Exception as e:
        pass

def generar_frames():
    """Generador de frames para el streaming de video"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(img_rgb)
        
        # Información en pantalla
        if molecular_system.current_molecule:
            molecule_name = molecular_system.current_molecule['name']
            cv2.putText(frame, f"Molécula: {molecule_name}", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, "Gestos: Puño=Mover | Dedos=Zoom | Mano=Rotar", 
                   (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Procesar detección de manos
        if resultado.multi_hand_landmarks:
            for hand_landmarks in resultado.multi_hand_landmarks:
                mp_dibujo.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                molecular_system.update_molecule_position(hand_landmarks, frame.shape)
        
        # Dibujar molécula si existe
        if molecular_system.current_molecule:
            mol_data = molecular_system.molecules['primary']
            if mol_data['active'] or molecular_system.current_molecule:
                alpha = 1.0 if mol_data['active'] else 0.8
                dibujar_overlay_molecular(
                    frame, 
                    molecular_system.current_molecule['image'], 
                    mol_data['pos'], 
                    mol_data['scale'], 
                    alpha
                )
                
                # Indicador de molécula activa
                if mol_data['active']:
                    pos = mol_data['pos']
                    cv2.circle(frame, (pos[0] + 100, pos[1] + 100), 80, (0, 255, 255), 3)
        
        # Codificar frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

# ========== RUTAS FLASK ==========

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/camara')
def camara():
    """Stream de video con detección molecular"""
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/load_molecule', methods=['POST'])
def load_molecule():
    """Carga una molécula desde SMILES"""
    try:
        data = request.json
        print(f"🔬 Datos recibidos: {data}")  # Debug
        
        smiles = data.get('smiles', '').strip()
        name = data.get('name', 'Molécula Sin Nombre').strip()
        
        print(f"🧬 Cargando: {name} con SMILES: {smiles}")  # Debug
        
        if not smiles:
            return jsonify({'success': False, 'error': 'SMILES requerido'})
        
        result = molecular_system.load_molecule_from_smiles(smiles, name)
        print(f"✅ Resultado: {result}")  # Debug
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error en load_molecule: {str(e)}")  # Debug
        return jsonify({'success': False, 'error': f'Error del servidor: {str(e)}'})

@app.route('/validate_smiles', methods=['POST'])
def validate_smiles():
    """Valida un código SMILES"""
    try:
        data = request.json
        print(f"🔍 Validando SMILES: {data}")  # Debug
        
        smiles = data.get('smiles', '').strip()
        
        if not smiles:
            return jsonify({'valid': False, 'error': 'SMILES vacío'})
        
        result = molecular_system.validate_smiles(smiles)
        print(f"📋 Resultado validación: {result}")  # Debug
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error en validate_smiles: {str(e)}")  # Debug
        return jsonify({'valid': False, 'error': f'Error del servidor: {str(e)}'})

@app.route('/molecular_data')
def molecular_data():
    """Obtiene datos moleculares actuales"""
    try:
        data = molecular_system.get_molecular_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'Error obteniendo datos: {str(e)}'})

@app.route('/current_molecule')
def current_molecule():
    """Obtiene información de la molécula actual"""
    try:
        if molecular_system.current_molecule:
            return jsonify({
                'has_molecule': True,
                'name': molecular_system.current_molecule['name'],
                'smiles': molecular_system.current_molecule['smiles'],
                'properties': molecular_system.current_molecule['properties']
            })
        else:
            return jsonify({'has_molecule': False})
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/molecular_properties', methods=['POST'])
def molecular_properties():
    """Obtiene propiedades de una molécula por SMILES"""
    try:
        data = request.json
        smiles = data.get('smiles', '').strip()
        
        if not smiles:
            return jsonify({'success': False, 'error': 'SMILES requerido'})
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({'success': False, 'error': 'SMILES inválido'})
        
        properties = {
            'molecular_weight': round(rdMolDescriptors.CalcExactMolWt(mol), 2),
            'logp': round(rdMolDescriptors.Crippen.MolLogP(mol), 2),
            'hbd': rdMolDescriptors.CalcNumHBD(mol),
            'hba': rdMolDescriptors.CalcNumHBA(mol),
            'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'num_atoms': mol.GetNumAtoms()
        }
        
        return jsonify({'success': True, 'properties': properties})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})

@app.route('/reset_molecules', methods=['POST'])
def reset_molecules():
    """Reinicia el sistema molecular"""
    try:
        molecular_system.reset()
        return jsonify({'success': True, 'message': 'Sistema molecular reiniciado'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})

@app.route('/capture_interaction', methods=['POST'])
def capture_interaction():
    """Captura el estado actual de interacción"""
    try:
        # Aquí podrías guardar una imagen o datos del estado actual
        timestamp = int(time.time())
        capture_data = {
            'timestamp': timestamp,
            'molecule': molecular_system.current_molecule['name'] if molecular_system.current_molecule else None,
            'position': molecular_system.molecules['primary']['pos'],
            'scale': molecular_system.molecules['primary']['scale'],
            'rotation': {
                'x': molecular_system.molecules['primary']['rotation_x'],
                'y': molecular_system.molecules['primary']['rotation_y'],
                'z': molecular_system.molecules['primary']['rotation_z']
            }
        }
        
        return jsonify({
            'success': True, 
            'message': 'Interacción capturada',
            'data': capture_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})

# ========== MANEJO DE ERRORES ==========

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

# ========== IMPORTACIÓN FALTANTE ==========
import time

if __name__ == '__main__':
    print("🧬 Iniciando Visualizador Molecular AR...")
    print("📡 Servidor corriendo en: http://localhost:5000")
    print("📹 Asegúrate de que tu cámara esté conectada")
    print("🔬 Sistema RDKit inicializado correctamente")
    
    app.run(debug=True, host='0.0.0.0', port=5000)