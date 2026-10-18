"""
Questao 2 - Realce de imagens: power-law (gamma) e equalizacao de
histograma, aplicados a car.png, crowd.png e university.png.
"""
import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fontes maiores: as figuras serao exibidas em largura de UMA coluna no
# relatorio (twocolumn), entao o texto precisa ser maior na imagem fonte
# para continuar legivel depois de reduzida.
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "figure.titlesize": 24,
})

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figs")
os.makedirs(FIG_DIR, exist_ok=True)

IMAGENS = ["car.png", "crowd.png", "university.png"]

# gamma < 1 clareia (bom p/ imagens escuras); gamma > 1 escurece/aumenta
# contraste (bom p/ imagens estouradas/claras). Valores escolhidos apos
# inspecionar cada imagem.
GAMMAS = {
    "car.png": [0.6, 3.0, 5.0],          # imagem muito clara/lavada -> gamma>1
    "crowd.png": [0.3, 0.5, 2.0],        # imagem escura -> gamma<1
    "university.png": [0.2, 0.4, 2.0],   # imagem muito escura -> gamma<1
}

MELHOR_GAMMA = {
    "car.png": 3.0,
    "crowd.png": 0.3,
    "university.png": 0.4,
}


def power_law(img_gray: np.ndarray, gamma: float) -> np.ndarray:
    """Correcao gamma: s = c * r^gamma, com r normalizado em [0,1].
    Equivalente ao par cvPow + cvConvertScaleAbs do OpenCV antigo."""
    normalizada = img_gray.astype(np.float64) / 255.0
    realcada = cv2.pow(normalizada, gamma)
    c = 255.0  # constante de escala (r,s em [0,1] -> [0,255])
    saida = cv2.convertScaleAbs(realcada, alpha=c)
    return saida


def salvar(nome, img):
    path = os.path.join(FIG_DIR, nome)
    cv2.imwrite(path, img)
    print(f"salvo: {path}")


def parte_power_law():
    for nome in IMAGENS:
        img = cv2.imread(os.path.join(DATA_DIR, nome), cv2.IMREAD_GRAYSCALE)
        base = os.path.splitext(nome)[0]
        salvar(f"{base}_original.png", img)

        resultados = [img]
        legendas = ["Original"]
        for gamma in GAMMAS[nome]:
            r = power_law(img, gamma)
            salvar(f"{base}_gamma_{str(gamma).replace('.', '_')}.png", r)
            resultados.append(r)
            legendas.append(f"gamma={gamma}")

        melhor = power_law(img, MELHOR_GAMMA[nome])
        salvar(f"{base}_melhor_gamma.png", melhor)

        # figura comparativa (original + todos os gammas testados), em grade
        # 2x2 (em vez de 1x4) para ficar legivel quando reduzida a largura
        # de uma coluna no relatorio twocolumn.
        fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        for ax, im, leg in zip(axes.flat, resultados, legendas):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(leg, fontsize=22)
            ax.axis("off")
        fig.suptitle(f"Power-law (gamma) - {nome}", fontsize=26)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"{base}_gamma_comparativo.png"), dpi=130)
        plt.close(fig)


def cdf_normalizada(hist):
    cdf = hist.cumsum()
    return cdf / cdf[-1]


def parte_equalizacao():
    equalizadas = {}
    for nome in IMAGENS:
        img = cv2.imread(os.path.join(DATA_DIR, nome), cv2.IMREAD_GRAYSCALE)
        base = os.path.splitext(nome)[0]
        eq = cv2.equalizeHist(img)
        equalizadas[nome] = eq
        salvar(f"{base}_equalizada.png", eq)

    # figura com as 3 imagens equalizadas, empilhadas verticalmente (em vez
    # de lado a lado) para ficar legivel em largura de uma coluna.
    fig, axes = plt.subplots(3, 1, figsize=(6, 13))
    for ax, nome in zip(axes, IMAGENS):
        ax.imshow(equalizadas[nome], cmap="gray", vmin=0, vmax=255)
        ax.set_title(nome, fontsize=20)
        ax.axis("off")
    fig.suptitle("Equalizacao de histograma", fontsize=24)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "todas_equalizadas.png"), dpi=130)
    plt.close(fig)

    # histograma + CDF antes/depois para university.png (imagem mais escura,
    # efeito mais evidente)
    alvo = "university.png"
    base = os.path.splitext(alvo)[0]
    img = cv2.imread(os.path.join(DATA_DIR, alvo), cv2.IMREAD_GRAYSCALE)
    eq = equalizadas[alvo]

    hist_antes, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    hist_depois, _ = np.histogram(eq.flatten(), bins=256, range=(0, 256))
    cdf_antes = cdf_normalizada(hist_antes)
    cdf_depois = cdf_normalizada(hist_depois)

    # figsize quase quadrado (em vez de 11x8) para reduzir melhor quando
    # exibido em largura de uma coluna; fontes maiores pelo mesmo motivo.
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))

    axes[0, 0].bar(range(256), hist_antes, width=1.0, color="steelblue")
    axes[0, 0].set_title("Histograma antes", fontsize=20)
    axes[0, 0].set_xlabel("Nivel de cinza", fontsize=17)
    axes[0, 0].set_ylabel("Numero de pixels", fontsize=17)

    axes[0, 1].bar(range(256), hist_depois, width=1.0, color="darkorange")
    axes[0, 1].set_title("Histograma depois", fontsize=20)
    axes[0, 1].set_xlabel("Nivel de cinza", fontsize=17)
    axes[0, 1].set_ylabel("Numero de pixels", fontsize=17)

    axes[1, 0].plot(cdf_antes, color="steelblue")
    axes[1, 0].set_title("CDF antes", fontsize=20)
    axes[1, 0].set_xlabel("Nivel de cinza", fontsize=17)
    axes[1, 0].set_ylabel("CDF normalizada", fontsize=17)
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(cdf_depois, color="darkorange")
    axes[1, 1].set_title("CDF depois", fontsize=20)
    axes[1, 1].set_xlabel("Nivel de cinza", fontsize=17)
    axes[1, 1].set_ylabel("CDF normalizada", fontsize=17)
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_ylim(0, 1)

    fig.suptitle(f"Histograma e CDF - {alvo}", fontsize=24)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{base}_hist_cdf.png"), dpi=130)
    plt.close(fig)

    # comparativo lado a lado: original vs equalizada, para a imagem alvo
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original", fontsize=22)
    axes[0].axis("off")
    axes[1].imshow(eq, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Equalizada", fontsize=22)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{base}_antes_depois.png"), dpi=130)
    plt.close(fig)


def main():
    parte_power_law()
    parte_equalizacao()
    print("Questao 2 concluida.")


if __name__ == "__main__":
    main()
