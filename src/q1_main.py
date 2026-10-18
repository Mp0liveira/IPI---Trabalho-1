"""
Script principal da Questao 1.
Gera as imagens de resultado (TAM2 x2/x8, TAMM x2, SUPERRES) usadas no
relatorio, a partir das fotos fruit1.jpg / fruit2.jpg (substitutas de
FRUIT1.jpg / FRUIT2.jpg).
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tam2 import TAM2
from tamm import TAMM
from superres import SUPERRES

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def salvar(nome, img):
    path = os.path.join(FIG_DIR, nome)
    cv2.imwrite(path, img)
    print(f"salvo: {path}  shape={img.shape}")


def main():
    fruit1 = cv2.imread(os.path.join(DATA_DIR, "fruit1.jpg"))
    fruit2 = cv2.imread(os.path.join(DATA_DIR, "fruit2.jpg"))

    salvar("fruit1_original.png", fruit1)
    salvar("fruit2_original.png", fruit2)

    # --- 1.1 TAM2 ---
    fruit1_tam2_x2 = TAM2(fruit1, 2)
    fruit1_tam2_x8 = TAM2(fruit1, 8)
    salvar("fruit1_tam2_x2.png", fruit1_tam2_x2)
    salvar("fruit1_tam2_x8.png", fruit1_tam2_x8)

    # crop de uma regiao pequena para evidenciar o efeito de blocos (TAM2)
    crop = fruit1[80:110, 80:110]
    crop_tam2_x2 = TAM2(crop, 2)
    crop_tam2_x8 = TAM2(crop, 8)
    salvar("fruit1_crop_original.png", crop)
    salvar("fruit1_crop_tam2_x2.png", crop_tam2_x2)
    salvar("fruit1_crop_tam2_x8.png", crop_tam2_x8)

    # --- 1.2 TAMM ---
    fruit1_tamm_x2 = TAMM(fruit1)
    salvar("fruit1_tamm_x2.png", fruit1_tamm_x2)
    crop_tamm_x2 = TAMM(crop)
    salvar("fruit1_crop_tamm_x2.png", crop_tamm_x2)

    # TAMM aplicada 3x (2*2*2=8) para comparar com TAM2 x8 no mesmo fator de
    # aumento -- a diferenca entre repeticao (TAM2) e media (TAMM) fica muito
    # mais visivel nesse fator mais extremo do que em x2.
    crop_tamm_x8 = TAMM(TAMM(crop_tamm_x2))
    salvar("fruit1_crop_tamm_x8.png", crop_tamm_x8)

    # --- 1.3 SUPERRES ---
    combinada = SUPERRES(fruit1, fruit2)
    salvar("fruit_superres.png", combinada)

    print("Questao 1 concluida.")


if __name__ == "__main__":
    main()
