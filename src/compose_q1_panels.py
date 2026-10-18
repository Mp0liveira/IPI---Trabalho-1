"""
Compoe figuras de painel unico (varias imagens lado a lado, com titulos)
para a Questao 1, no mesmo estilo das figuras comparativas ja usadas na
Questao 2 (evita depender do pacote 'subcaption' do LaTeX, que entraria em
conflito com o estilo de legendas customizado do template IEEE/CVPR).
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figs")


def ler_rgb(nome):
    img = cv2.imread(os.path.join(FIG_DIR, nome))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def painel(nomes_arquivos, titulos, nome_saida, figsize=None, suptitle=None):
    # Estas figuras sao exibidas em largura de UMA coluna no relatorio
    # (twocolumn), reduzidas por um fator de ~2-3x em relacao ao tamanho
    # gerado aqui. Por isso usamos fontes um pouco maiores que o padrao, e
    # titulos curtos (detalhes ficam na legenda em LaTeX) para evitar que
    # o texto de um subplot invada o espaco do vizinho.
    n = len(nomes_arquivos)
    if figsize is None:
        figsize = (3.3 * n, 3.6)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, nome, titulo in zip(axes, nomes_arquivos, titulos):
        ax.imshow(ler_rgb(nome))
        ax.set_title(titulo, fontsize=16)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=18)
    fig.tight_layout(w_pad=1.5)
    out = os.path.join(FIG_DIR, nome_saida)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("salvo:", out)


def main():
    painel(
        ["fruit1_original.png", "fruit2_original.png"],
        ["fruit1.jpg", "fruit2.jpg"],
        "painel_originais.png",
    )

    painel(
        ["fruit1_original.png", "fruit1_tam2_x2.png", "fruit1_tam2_x8.png"],
        ["Original", "TAM2 x2", "TAM2 x8"],
        "painel_tam2.png",
    )

    # Nota: nao comparamos aqui "recorte original" com "TAM2", pois ao
    # exibir uma imagem pequena (30x30) numa caixa maior, o matplotlib faz
    # exatamente a mesma ampliacao por vizinho mais proximo que o TAM2 faz
    # -- os dois ficariam visualmente identicos, nao por erro, mas porque
    # matematicamente SAO a mesma operacao. Por isso comparamos TAM2 contra
    # TAMM diretamente (que usam dados realmente diferentes).
    painel(
        ["fruit1_crop_tam2_x8.png", "fruit1_crop_tamm_x8.png"],
        ["TAM2 x8 (repeticao)", "TAMM x8 (media, 3x)"],
        "painel_tam2_zoom.png",
        figsize=(9, 5),
    )

    painel(
        ["fruit1_original.png", "fruit1_tam2_x2.png", "fruit1_tamm_x2.png"],
        ["Original", "TAM2 x2", "TAMM x2"],
        "painel_tam2_vs_tamm.png",
    )

    painel(
        ["fruit1_crop_tam2_x2.png", "fruit1_crop_tamm_x2.png"],
        ["TAM2 x2 (repeticao)", "TAMM x2 (media)"],
        "painel_tam2_vs_tamm_zoom.png",
        figsize=(9, 5),
    )

    print("Paineis da Questao 1 gerados.")


if __name__ == "__main__":
    main()
