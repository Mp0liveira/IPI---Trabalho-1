"""
Questao 1.3 - SUPERRES
Combina duas imagens HxW do mesmo tamanho em uma unica imagem (2H)x(2W):

  - img1 ocupa as posicoes de (linha, coluna) IMPARES (indexacao do
    MATLAB, comecando em 1). Em indexacao 0-based (Python) isso
    corresponde as posicoes (linha PAR, coluna PAR).
  - img2 ocupa as posicoes (linha, coluna) PARES (MATLAB) == (linha
    IMPAR, coluna IMPAR) em indexacao 0-based.

Exemplo do enunciado (coordenadas 1-based, MATLAB):
    img1(1,1) -> final(1,1)      img2(1,1) -> final(2,2)
    img1(1,2) -> final(1,3)      img2(1,2) -> final(2,4)
    img1(2,2) -> final(3,3)      img2(2,2) -> final(4,4)

O enunciado nao especifica o que colocar nas posicoes restantes do
"tabuleiro de xadrez" (linha par/coluna impar e linha impar/coluna
par). Adotamos a solucao natural de preenchimento por interpolacao:
cada uma dessas posicoes tem exatamente os 4 vizinhos diretos (cima,
baixo, esquerda, direita) ja conhecidos -- dois vindos de img1 e dois
de img2 -- entao preenchemos com a media desses vizinhos (ou dos que
existirem, nas bordas), seguindo o mesmo espirito de "preencher com
media quando possivel" usado na funcao TAMM (questao 1.2).
"""
import numpy as np


def _round_half_up(x: np.ndarray) -> np.ndarray:
    return np.floor(x + 0.5)


def SUPERRES(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Combina img1 e img2 (mesmo tamanho HxW, RGB) em uma imagem (2H)x(2W).

    Parametros
    ----------
    img1, img2 : np.ndarray
        Imagens coloridas (H, W, 3), mesmo tamanho.

    Retorna
    -------
    np.ndarray
        Imagem combinada (2H, 2W, 3), dtype uint8.
    """
    if img1.shape != img2.shape:
        raise ValueError("img1 e img2 devem ter o mesmo tamanho.")

    h, w, c = img1.shape
    out = np.zeros((2 * h, 2 * w, c), dtype=np.float64)
    preenchido = np.zeros((2 * h, 2 * w), dtype=bool)

    # img1 -> posicoes (linha par, coluna par) em 0-based
    #       = posicoes (linha impar, coluna impar) em 1-based (MATLAB)
    out[0::2, 0::2, :] = img1
    preenchido[0::2, 0::2] = True

    # img2 -> posicoes (linha impar, coluna impar) em 0-based
    #       = posicoes (linha par, coluna par) em 1-based (MATLAB)
    out[1::2, 1::2, :] = img2
    preenchido[1::2, 1::2] = True

    # Posicoes restantes (linha par/col impar e linha impar/col par):
    # preenche com a media dos vizinhos diretos ja conhecidos.
    H2, W2 = 2 * h, 2 * w
    rows, cols = np.where(~preenchido)
    for r, cidx in zip(rows, cols):
        vizinhos = []
        if r - 1 >= 0:
            vizinhos.append(out[r - 1, cidx, :])
        if r + 1 < H2:
            vizinhos.append(out[r + 1, cidx, :])
        if cidx - 1 >= 0:
            vizinhos.append(out[r, cidx - 1, :])
        if cidx + 1 < W2:
            vizinhos.append(out[r, cidx + 1, :])
        media = _round_half_up(np.mean(np.stack(vizinhos, axis=0), axis=0))
        out[r, cidx, :] = media

    return np.clip(out, 0, 255).astype(np.uint8)
