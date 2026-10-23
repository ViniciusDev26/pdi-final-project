"""Biblioteca de PDI compartilhada entre o treinamento e a API.

Todo pré-processamento usado no treino deve morar aqui, para que a API
aplique exatamente as mesmas transformações na inferência.
"""

from pdi.io import load_rgb

__all__ = ["load_rgb"]
