import os
import cv2
import face_recognition

# Directorio que contiene las carpetas de imágenes de cada persona
directory = "Caras"

# Ruta completa al directorio de imágenes
directory_path = os.path.join(os.path.dirname(__file__), directory)

# Diccionario para almacenar las codificaciones faciales de cada persona
known_face_encodings = {}
known_face_names = []

# Recorrer cada carpeta en el directorio
for subdir, dirs, files in os.walk(directory_path):
    for file in files:
        # Cargar la imagen de la persona
        image_path = os.path.join(subdir, file)
        person_name = os.path.basename(subdir)
        image = face_recognition.load_image_file(image_path)

        # Codificar la cara en la imagen
        encoding = face_recognition.face_encodings(image)[0]

        # Guardar la codificación facial junto con el nombre de la persona
        if person_name not in known_face_encodings:
            known_face_encodings[person_name] = []
            known_face_names.append(person_name)
        known_face_encodings[person_name].append(encoding)

# Iniciar la captura de vídeo desde la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer un fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el fotograma de BGR a RGB (necesario para face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar las caras en el fotograma
    face_locations = face_recognition.face_locations(rgb_frame)
    if face_locations:
        # Obtener las codificaciones faciales de las caras detectadas
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Comparar las codificaciones faciales de las caras detectadas con las imágenes conocidas
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            name = "Desconocido"

            # Comparar la codificación facial con las imágenes conocidas
            for known_name, known_encodings in known_face_encodings.items():
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                if True in matches:
                    name = known_name
                    break

            # Dibujar un cuadro alrededor de la cara y mostrar el nombre
            color = (0, 255, 0) if name != "Desconocido" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Reconocimiento facial en tiempo real', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
