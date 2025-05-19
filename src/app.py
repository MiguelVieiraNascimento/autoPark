import cv2
import numpy as np
from paddleocr import PaddleOCR
from datetime import datetime
import pandas as pd

# Inicializa o PaddleOCR com configurações otimizadas para placas
ocr = PaddleOCR(use_angle_cls=True, lang='en',
                rec_model_dir='en_PP-OCRv3_rec',
                det_model_dir='en_PP-OCRv3_det',
                cls_model_dir='ch_ppocr_mobile_v2.0_cls')


def carregar_placas_planilha(caminho_arquivo):
    try:
        df = pd.read_excel(caminho_arquivo)
        #todo mudar o nome da coluna para 'Placas', porque tá 'Placa'
        if 'Placa' not in df.columns:
            raise ValueError("A planilha deve conter uma coluna chamada 'Placa'")
        placas = df['Placa'].dropna().astype(str).str.upper().str.strip().tolist()
        print(f"[DEBUG] Placas carregadas da planilha: {placas}")
        return placas
    except Exception as e:
        print(f"[ERRO] Falha ao carregar placas da planilha: {e}")
        return []


def process_video():
    placas_cadastradas = carregar_placas_planilha("DBplacas.xlsx")
    cap = cv2.VideoCapture(0)  # Utiliza a câmera padrão

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    # Configurações da câmera para melhor detecção
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o vídeo.")
            break

        # Pré-processamento para melhorar a detecção
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # Converter para RGB para o PaddleOCR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecção de texto
        results = ocr.ocr(rgb_frame, cls=True)

        detected_text = ""
        plate_boxes = []

        if results and results[0]:
            for line in results[0]:
                if line:
                    text = line[1][0]
                    box = line[0]

                    # Filtro para textos que parecem placas (exemplo para placas brasileiras)
                    if len(text) >= 6 and any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
                        detected_text = text.upper()
                        print(f"[DEBUG] Texto detectado: '{detected_text}'")
                        print(f"[DEBUG] Placas cadastradas: {placas_cadastradas}")

                        plate_boxes.append(np.int32(box))

                        # Desenha a caixa ao redor da placa
                        cv2.polylines(frame, [np.int32(box)], isClosed=True,
                                      color=(0, 255, 0), thickness=2)

                        # Mostra o texto detectado
                        cv2.putText(frame, detected_text,
                                    org=(int(box[0][0]), int(box[0][1]) - 10),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.8,
                                    color=(0, 255, 0), thickness=2)

        # Informações na tela
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if detected_text:
            if detected_text in placas_cadastradas:
                mensagem = f"Placa AUTORIZADA: {detected_text}"
                cor = (0, 255, 0)  # verde
            else:
                mensagem = f"Placa NAO cadastrada: {detected_text}"
                cor = (0, 0, 255)  # vermelho

            cv2.putText(frame, mensagem, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        else:
            cv2.putText(frame, "Nenhuma placa detectada", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Mostra a imagem
        cv2.imshow('Reconhecimento de Placas', cv2.resize(frame, (800, 600)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video()
