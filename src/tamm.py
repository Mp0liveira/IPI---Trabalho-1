"""
Questao 1.2 - TAMM
Aumenta uma imagem RGB em 2 vezes usando MEDIA entre pixels vizinhos
(quando possivel), em vez de repeticao pura (TAM2).

Conforme o enunciado, o preenchimento ocorre em duas passadas:
  1) "Pixels vermelhos": preenchem as linhas pares (0-indexed) do
     resultado, que ja contem os pixels originais (colocados nas
     colunas pares). Cada pixel novo (coluna impar) recebe a media
     entre o pixel original a esquerda e a direita. Na ultima coluna,
     onde nao ha vizinho a direita, repete-se o valor anterior.
  2) "Pixels azuis": preenchem as linhas impares (totalmente vazias),
     fazendo a media vertical entre a linha de cima e a linha de baixo
     (ja preenchidas no passo 1). Na ultima linha, onde nao ha linha
     de baixo, repete-se a linha de cima.

Exemplo do enunciado (1 canal):
    Original:            10 12
                          5  7

    TAMM:   10 11 12 12
             8  9 10 10
             5  6  7  7
             5  6  7  7
"""
import numpy as np


def _round_half_up(x: np.ndarray) -> np.ndarray:
    """Arredondamento '0.5 sempre para cima' (np.round usa banker's
    rounding, que arredondaria 8.5 -> 8; o exemplo do enunciado espera
    8.5 -> 9)."""
    return np.floor(x + 0.5)


def TAMM(img: np.ndarray) -> np.ndarray:
    """
    Aumenta a imagem `img` (RGB) em 2 vezes usando interpolacao por media.

    Parametros
    ----------
    img : np.ndarray
        Imagem colorida (H, W, 3), dtype uint8.

    Retorna
    -------
    np.ndarray
        Imagem aumentada (2H, 2W, 3), dtype uint8.
    """
    img = img.astype(np.float64)
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1

    out = np.zeros((2 * h, 2 * w, c), dtype=np.float64)

    # Coloca os pixels originais nas posicoes (linha par, coluna par).
    out[0::2, 0::2, :] = img

    # --- Passo 1 (pixels "vermelhos"): preenche colunas impares das
    # linhas pares, fazendo a media horizontal quando possivel. ---
    for j in range(1, 2 * w, 2):
        j_prev = j - 1          # coluna par (original) a esquerda
        j_next = j + 1          # coluna par (original) a direita
        if j_next < 2 * w:
            out[0::2, j, :] = _round_half_up((out[0::2, j_prev, :] + out[0::2, j_next, :]) / 2.0)
        else:
            # ultima coluna: sem vizinho a direita, repete o anterior
            out[0::2, j, :] = out[0::2, j_prev, :]

    # --- Passo 2 (pixels "azuis"): preenche as linhas impares
    # (totalmente vazias), fazendo a media vertical quando possivel. ---
    for i in range(1, 2 * h, 2):
        i_prev = i - 1           # linha acima, ja preenchida no passo 1
        i_next = i + 1           # linha abaixo
        if i_next < 2 * h:
            out[i, :, :] = _round_half_up((out[i_prev, :, :] + out[i_next, :, :]) / 2.0)
        else:
            # ultima linha: sem linha abaixo, repete a linha acima
            out[i, :, :] = out[i_prev, :, :]

    return np.clip(out, 0, 255).astype(np.uint8)
