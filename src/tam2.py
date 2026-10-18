"""
Questao 1.1 - TAM2
Aumenta uma imagem RGB em N vezes (N par, N > 2) repetindo cada
linha e cada coluna, exatamente como no exemplo do enunciado:

    10 12          10 10 12 12
    5  7    -(x2)-> 10 10 12 12
                     5  5  7  7
                     5  5  7  7

Estrategia (conforme a dica do enunciado): implementamos uma funcao
auxiliar que aumenta a imagem em 2 vezes (aumenta2) e a chamamos
repetidamente dentro de TAM2 ate atingir o fator N desejado.
"""
import numpy as np


def aumenta2(img: np.ndarray) -> np.ndarray:
    """Duplica cada linha e cada coluna da imagem (fator de aumento = 2)."""
    dobrado_linhas = np.repeat(img, 2, axis=0)
    dobrado = np.repeat(dobrado_linhas, 2, axis=1)
    return dobrado


def repete_fator(img: np.ndarray, fator: int) -> np.ndarray:
    """Repete cada linha/coluna 'fator' vezes (usado para o fator residual
    quando N nao e uma potencia de 2 pura)."""
    rep_linhas = np.repeat(img, fator, axis=0)
    rep = np.repeat(rep_linhas, fator, axis=1)
    return rep


def TAM2(img: np.ndarray, n: int) -> np.ndarray:
    """
    Aumenta a imagem `img` (RGB) em `n` vezes nas linhas e colunas,
    repetindo os pixels da linha/coluna anterior.

    Parametros
    ----------
    img : np.ndarray
        Imagem colorida (H, W, 3).
    n : int
        Fator de aumento, par e >= 2 (ex.: 2, 4, 8, 16, ...). O enunciado
        pede explicitamente os casos n=2 e n=8 como demonstracao.

    Retorna
    -------
    np.ndarray
        Imagem aumentada (H*n, W*n, 3).
    """
    if n < 2 or n % 2 != 0:
        raise ValueError("n deve ser um numero par maior ou igual a 2 (ex.: 2, 4, 8).")

    resultado = img.copy()
    fator_restante = n

    # Chama a funcao de dobrar (aumenta2) repetidamente enquanto o fator
    # residual for par -> cobre exatamente os casos pedidos (2 e 8, e
    # qualquer potencia de 2: 4, 16, ...).
    while fator_restante % 2 == 0:
        resultado = aumenta2(resultado)
        fator_restante //= 2

    # Caso n nao seja uma potencia pura de 2 (ex.: 6, 12), aplica o fator
    # impar restante diretamente por repeticao (equivalente, mas em um
    # unico passo).
    if fator_restante > 1:
        resultado = repete_fator(resultado, fator_restante)

    return resultado
