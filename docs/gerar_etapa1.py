"""Gera o PDF da Etapa 1 (Equipe, Público-Alvo e Proposta).

Uso: python3 docs/gerar_etapa1.py
Monta um HTML e o converte em PDF com o Google Chrome (headless).
Trechos destacados em amarelo no documento são dados a conferir/preencher.
"""

import html
import re
import subprocess
from pathlib import Path

PASTA = Path(__file__).parent
SAIDA_HTML = PASTA / "PDI_Etapa1_Equipe_e_Proposta.html"
SAIDA_PDF = PASTA / "PDI_Etapa1_Equipe_e_Proposta.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

EQUIPE = [
    ("Carlos Vinicius Bezerra Lopes (representante)", "2020808",
     "Coordenação e entregas formais; arquitetura do sistema; desenvolvimento da API, da "
     "interface (front-end) e da infraestrutura do protótipo; integração dos modelos treinados."),
    ("Renan Welling Barroso Aguiar", "2214584",
     "Pesquisa bibliográfica; pré-processamento (remoção de pelos, normalização de cor, "
     "realce), segmentação das lesões e extração de características (ABCD, cor e textura)."),
    ("Guilherme Almeida", "2410535",
     "Treinamento dos modelos (classificadores tradicionais e rede neural); definição das "
     "métricas; avaliação e análise dos resultados."),
]


# ---------------------------------------------------------------- helpers

def rico(texto):
    """Converte marcação simples em HTML: **negrito** e [[destaque amarelo]]."""
    texto = html.escape(texto, quote=False)
    texto = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", texto)
    return re.sub(r"\[\[(.+?)\]\]", r"<mark>\1</mark>", texto)


def p(texto):
    return f"<p>{rico(texto)}</p>"


def topicos(itens, numerado=False):
    tag = "ol" if numerado else "ul"
    return f"<{tag}>" + "".join(f"<li>{rico(i)}</li>" for i in itens) + f"</{tag}>"


def secao(texto):
    return f'<h2 class="secao">{html.escape(texto)}</h2>'


def subtitulo(texto):
    return f"<h3>{html.escape(texto)}</h3>"


def tabela(cabecalho, linhas, larguras):
    cols = "".join(f'<col style="width:{w}%">' for w in larguras)
    th = "".join(f"<th>{html.escape(c)}</th>" for c in cabecalho)
    trs = "".join("<tr>" + "".join(f"<td>{rico(v)}</td>" for v in linha) + "</tr>"
                  for linha in linhas)
    return f'<table class="dados"><colgroup>{cols}</colgroup><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


# ---------------------------------------------------------------- conteúdo

def cabecalho():
    return """
<div class="faixa">Universidade de Fortaleza — Unifor</div>
<table class="cabecalho">
  <tr><th>Professora:</th><td>Lyndainês Santos</td>
      <th>Documento:</th><td>Composição da Equipe, Público-Alvo e Proposta</td></tr>
  <tr><th>Disciplina:</th><td>Processamento Digital de Imagens (P.D.I.)</td>
      <th>Etapa:</th><td>1</td></tr>
</table>
<h1>Classificação de lesões de pele em imagens dermatoscópicas para apoio à triagem do
melanoma: uma comparação entre técnicas tradicionais de PDI e aprendizado profundo</h1>
<p class="local">Fortaleza/CE, outubro de 2026</p>
"""


def secao_equipe():
    return (
        secao("Composição da Equipe")
        + tabela(["Nome completo", "Matrícula", "Responsabilidade inicial"], EQUIPE, [32, 14, 54])
        + p("**Representante da equipe:** Carlos Vinicius Bezerra Lopes (matrícula 2020808), "
            "responsável pelas entregas formais.")
        + p("A **documentação do projeto** é responsabilidade compartilhada por todos os "
            "integrantes, cada um redigindo a parte correspondente às suas atividades. "
            "A divisão poderá ser ajustada ao longo do desenvolvimento.")
    )


def secao_publico():
    return (
        secao("Identificação do Público-Alvo")
        + subtitulo("a) Público-alvo")
        + p("O público-alvo são os **profissionais da Atenção Primária à Saúde (APS)** — médicos "
            "de família e comunidade, clínicos gerais e enfermeiros que atuam em Unidades Básicas "
            "de Saúde (UBS) — responsáveis pelo primeiro contato com pacientes que apresentam "
            "lesões de pele suspeitas. Esses profissionais precisam decidir, muitas vezes sem apoio "
            "de um especialista, quais lesões devem ser encaminhadas com prioridade ao dermatologista. "
            "Indiretamente, beneficiam-se os **pacientes atendidos pelo SUS**, especialmente aqueles "
            "com maior exposição solar (trabalhadores ao ar livre, pessoas de pele clara e idosos), "
            "que dependem de um encaminhamento correto para obter diagnóstico precoce.")
        + subtitulo("b) Evidências da relevância social")
        + topicos([
            "**O câncer de pele é o tipo de câncer mais frequente no Brasil.** Segundo a estimativa do "
            "INCA para o triênio 2026–2028, são esperados cerca de 781 mil casos novos de câncer por "
            "ano no país, dos quais 263.280 são de câncer de pele não melanoma — mais de 30% do total "
            "(INCA, 2026).",
            "**O melanoma é raro, porém letal.** Para 2026, o INCA estima 9.360 casos novos de melanoma "
            "no Brasil (INCA, 2026). Ele representa cerca de 5% dos cânceres de pele, mas responde por "
            "até 80% das mortes pela doença; quando detectado precocemente, as chances de cura superam "
            "90% (O POVO, 2026).",
            "**O Ceará e Fortaleza apresentam números expressivos.** Para 2026, são estimados 11.110 "
            "casos novos de câncer de pele não melanoma e 300 de melanoma no Ceará; somente em "
            "Fortaleza, são 1.920 casos de não melanoma e 100 de melanoma (INCA, 2026). Em 2024, a "
            "taxa de diagnósticos de câncer de pele no Ceará foi de 68,64 por 100 mil habitantes, "
            "acima da média nacional de 34,27 (SBD, 2026).",
            "**Há escassez e má distribuição de dermatologistas.** Cerca de 63,5% dos dermatologistas "
            "estão no Sudeste, que concentra 41,6% da população (YARAK; RUIZ; PISA, 2017). No SUS, "
            "agendar uma consulta dermatológica é cerca de 2,6 vezes mais difícil do que na saúde "
            "suplementar, e no Norte e Nordeste o início do tratamento do câncer de pele "
            "frequentemente ultrapassa 60 dias, contra até 30 dias no Sul e Sudeste (SBD, 2026).",
            "**A demanda chega primeiro à APS.** Queixas dermatológicas estão entre os motivos mais "
            "comuns de consulta na atenção primária, e quase 60% desses atendimentos são realizados "
            "por médicos não dermatologistas, o que gera encaminhamentos desnecessários e atrasos "
            "no diagnóstico de casos graves (YARAK; RUIZ; PISA, 2017).",
        ])
        + subtitulo("c) Delimitação do contexto de aplicação")
        + p("A solução será aplicada na **triagem de lesões de pele a partir de imagens de "
            "dermatoscopia** (fotografias ampliadas da lesão obtidas com dermatoscópio, inclusive "
            "modelos de baixo custo acopláveis a smartphones), no contexto das **UBS de Fortaleza/CE**. "
            "Localizada a cerca de 3° de latitude sul, próxima à linha do Equador, a capital "
            "cearense recebe radiação solar intensa ao longo de todo o ano, o que torna a "
            "exposição ultravioleta um fator de risco constante para a população.")
        + p("O sistema **não pretende substituir o diagnóstico médico nem a biópsia**: seu papel é "
            "indicar se uma lesão apresenta características visuais **suspeitas de malignidade**, "
            "auxiliando o profissional a priorizar encaminhamentos ao dermatologista.")
        + p("Para que o escopo seja compatível com o tempo da disciplina, o protótipo será "
            "desenvolvido e avaliado com uma base pública e rotulada de imagens dermatoscópicas "
            "(Melanoma Skin Cancer Dataset of 10000 Images), sem coleta de imagens de pacientes. "
            "A tarefa será **binária**: lesão benigna × melanoma (maligna). Reconhece-se como "
            "limitação que a base não informa o perfil dos pacientes (idade, sexo, tom de pele) nem "
            "a origem das imagens, de modo que o desempenho em uma população miscigenada como a "
            "brasileira não pode ser garantido; esse ponto será discutido na avaliação dos resultados.")
    )


def secao_proposta():
    return (
        secao("Proposta Inicial do Aplicativo")
        + subtitulo("Introdução")
        + p("No organismo humano, a maior parte das células se renova continuamente: elas se "
            "dividem, cumprem sua função e, ao fim de um ciclo de vida limitado, passam por um "
            "processo de morte celular programada, chamado **apoptose**. Esse equilíbrio pode ser "
            "rompido por mutações genéticas que fazem as células se multiplicarem de forma "
            "descontrolada, dando origem ao câncer. Na pele, um dos principais fatores que provocam "
            "essas mutações é a exposição à radiação ultravioleta (UV), seja a natural, proveniente "
            "do sol no dia a dia, seja a artificial, como a das câmaras de bronzeamento. Quando "
            "atinge os melanócitos — as células que produzem a melanina, pigmento que dá cor à "
            "pele —, esse dano pode originar o **melanoma**, a forma mais agressiva de câncer de pele.")
        + p("O câncer de pele é o tipo de câncer mais comum no Brasil e, apenas entre 2009 e 2019, "
            "causou mais de 30 mil mortes no país (SBD apud JORNAL OPÇÃO, 2020). A chance de cura "
            "depende diretamente da detecção precoce, mas o acesso ao dermatologista no SUS é "
            "limitado, sobretudo no Norte e Nordeste. Na prática, quem examina primeiro a lesão é o "
            "profissional da Atenção "
            "Primária, que nem sempre tem treinamento específico para diferenciar uma pinta comum "
            "de uma lesão maligna. Isso gera dois problemas: encaminhamentos desnecessários, que "
            "congestionam as filas, e casos graves que demoram a ser identificados.")
        + p("O diagnóstico dermatológico é essencialmente **visual**: o especialista avalia "
            "assimetria, irregularidade da borda, variedade de cores, tamanho e textura da lesão — "
            "a chamada regra ABCD (NACHBAR et al., 1994). Esses critérios podem ser medidos computacionalmente a partir "
            "da imagem, o que torna o Processamento Digital de Imagens (PDI) pertinente: técnicas "
            "de PDI permitem remover interferências (pelos, reflexos, variação de iluminação), "
            "isolar a lesão da pele ao redor e extrair medidas objetivas de forma, cor e textura, "
            "que alimentam um classificador. Além disso, redes neurais convolucionais já "
            "demonstraram desempenho comparável ao de dermatologistas na classificação de lesões "
            "de pele (ESTEVA et al., 2017), o que reforça o potencial da abordagem.")
        + p("Diante disso, é de grande importância dispor de meios para uma análise prévia das "
            "lesões de pele. Este projeto propõe uma ferramenta de **apoio à triagem**, baseada em "
            "PDI e aprendizado de máquina, capaz de diferenciar lesões benignas de melanomas: ela "
            "recebe uma imagem dermatoscópica, segmenta a lesão, calcula indicadores visuais "
            "interpretáveis e estima se a lesão é suspeita de malignidade, apresentando ao "
            "profissional não só o resultado, mas também as evidências visuais que o justificam.")
        + subtitulo("Objetivo Geral")
        + p("Desenvolver e avaliar um protótipo de apoio à triagem do melanoma que, a partir "
            "de imagens dermatoscópicas, **segmente a lesão e a classifique como benigna ou maligna (melanoma)**, "
            "comparando uma abordagem baseada em técnicas tradicionais de PDI (pré-processamento, "
            "segmentação e extração de características ABCD, de cor e de textura) com uma abordagem "
            "baseada em aprendizado profundo (redes neurais convolucionais).")
        + subtitulo("Articulação das três dimensões")
        + p("**Dimensão social.** O projeto responde a uma necessidade concreta dos profissionais "
            "da Atenção Primária e dos pacientes do SUS em Fortaleza/CE: identificar precocemente "
            "lesões de pele suspeitas em um cenário de alta incidência de câncer de pele, escassez "
            "de dermatologistas e longas filas de espera. A ferramenta busca qualificar os "
            "encaminhamentos, priorizando os casos mais graves e reduzindo os desnecessários.")
        + p("**Dimensão tecnológica.** Técnicas de PDI são o núcleo da solução: o pré-processamento "
            "remove pelos e corrige cor e contraste, a segmentação isola a lesão e a extração de "
            "características transforma a imagem em medidas objetivas de assimetria, borda, cor e "
            "textura. Essas informações alimentam classificadores e são apresentadas ao profissional "
            "em um protótipo, junto com a lesão segmentada e a probabilidade de malignidade.")
        + p("**Dimensão científica.** O projeto investiga se uma abordagem tradicional e "
            "interpretável de PDI alcança desempenho comparável ao de uma rede neural, qual é a "
            "contribuição de cada etapa de pré-processamento e se os modelos decidem com base na "
            "lesão ou em artefatos da imagem. As respostas serão obtidas por experimentos "
            "controlados e avaliadas com métricas quantitativas, descritas adiante.")
        + subtitulo("Objetivos Específicos")
        + topicos([
            "**Investigar e implementar técnicas de pré-processamento** — remoção de pelos por "
            "morfologia matemática, normalização de cor e realce de contraste — e avaliar seu efeito "
            "na qualidade das etapas seguintes.",
            "**Implementar e avaliar a segmentação automática das lesões**, por limiarização e "
            "operações morfológicas, comparando as máscaras obtidas com máscaras de referência "
            "anotadas manualmente pela equipe em um subconjunto de imagens.",
            "**Extrair características de forma, borda, cor e textura** inspiradas na regra ABCD e "
            "treinar classificadores tradicionais (SVM e Random Forest) para separar lesões benignas "
            "e malignas.",
            "**Treinar uma rede neural convolucional por transferência de aprendizado** e comparar "
            "seu desempenho com o dos classificadores tradicionais, utilizando as mesmas divisões de "
            "dados e métricas.",
            "**Analisar a interpretabilidade e a robustez dos modelos**, verificando com mapas de "
            "ativação (Grad-CAM) e experimentos de ablação se a decisão se baseia na lesão ou em "
            "artefatos da imagem (pelos, réguas, bordas escuras do dermatoscópio).",
            "**Desenvolver um protótipo** (aplicação web) que receba uma imagem e exiba a lesão "
            "segmentada, os indicadores visuais calculados e a probabilidade de malignidade.",
        ], numerado=True)
        + subtitulo("Informações visuais extraídas e técnicas de PDI a investigar")
        + p("A tabela abaixo relaciona as técnicas que a equipe pretende investigar, a informação "
            "visual que cada uma extrai e por que ela contribui para a solução.")
        + tabela(["Técnica", "Informação extraída / transformação", "Contribuição para a solução"], [
            ("Conversão entre espaços de cores (RGB → HSV, CIELab)",
             "Separação entre luminosidade e cromaticidade.",
             "Facilita separar lesão e pele e medir a variedade de cores (critério C da regra ABCD)."),
            ("Normalização de cor (Shades of Gray; FINLAYSON; TREZZI, 2004)",
             "Correção da cor da iluminação de cada dermatoscópio.",
             "Reduz diferenças entre equipamentos, tornando as medidas de cor comparáveis."),
            ("Histograma e realce de contraste (CLAHE)",
             "Distribuição de intensidades; realce local no canal L.",
             "Evidencia a borda e as estruturas internas da lesão em imagens pouco contrastadas."),
            ("Filtragem espacial (mediana, gaussiana)",
             "Suavização de ruído e pequenas texturas da pele.",
             "Melhora a estabilidade da limiarização e da detecção de bordas."),
            ("Operações morfológicas (black-hat, abertura, fechamento) + inpainting",
             "Detecção e remoção de pelos (método DullRazor; LEE et al., 1997); limpeza das máscaras.",
             "Pelos sobre a lesão distorcem borda, cor e textura; removê-los reduz falsos padrões."),
            ("Limiarização global e adaptativa (Otsu)",
             "Separação automática entre lesão e pele.",
             "Gera a máscara da lesão, base para todas as medidas de forma."),
            ("Segmentação e análise de componentes conexos",
             "Região da lesão, contorno e eixos principais.",
             "Isola a lesão e descarta artefatos como réguas e bordas escuras."),
            ("Detecção de bordas (Canny, Sobel)",
             "Gradiente ao longo do contorno.",
             "Mede se a borda é nítida ou difusa e sua irregularidade (critério B)."),
            ("Extração de características de forma",
             "Assimetria, compacidade, solidez, dimensão fractal, área relativa.",
             "Quantifica os critérios A (assimetria), B (borda) e D (dimensão)."),
            ("Extração de características de cor e textura (GLCM, LBP)",
             "Estatísticas de cor por canal; contraste, homogeneidade e padrões locais.",
             "Captura a variação de cores e a desorganização da estrutura, típicas de lesões malignas."),
            ("Classificação tradicional (SVM, Random Forest)",
             "Decisão benigno × maligno a partir das características.",
             "Modelo interpretável: é possível saber quais medidas pesaram na decisão."),
            ("Classificação por aprendizado profundo (CNN com transfer learning) e Grad-CAM",
             "Aprendizado automático das características; mapa das regiões relevantes.",
             "Referência de desempenho e permite verificar onde o modelo \"olha\" na imagem."),
        ], [27, 33, 40])
        + subtitulo("Base de dados")
        + p("Será utilizada a base pública **Melanoma Skin Cancer Dataset of 10000 Images**, "
            "disponibilizada no Kaggle sob licença CC0 (domínio público) (JAVID, 2022). A base contém "
            "**10.605 imagens dermatoscópicas coloridas de 300 × 300 pixels**, rotuladas em duas classes "
            "— **benigna** e **maligna** (melanoma) — e já divididas em treino (5.000 benignas e 4.605 "
            "malignas) e teste (500 de cada classe). Por ser praticamente balanceada, a base favorece "
            "a comparação entre modelos sem a necessidade de técnicas agressivas de reamostragem.")
        + p("Uma análise exploratória preliminar realizada pela equipe identificou dois cuidados "
            "metodológicos necessários:")
        + topicos([
            "**Imagens duplicadas:** foram encontrados 22 pares de arquivos idênticos, sete deles "
            "presentes ao mesmo tempo no treino e no teste. Esses casos serão removidos para que o "
            "desempenho não seja superestimado.",
            "**Viés de aquisição:** a borda escura circular produzida por alguns dermatoscópios "
            "(vinheta) aparece em cerca de 31% das imagens malignas de treino e 43% das de teste, "
            "mas em apenas 0,6% das benignas. Um modelo pode aprender a associar essa borda ao "
            "melanoma sem analisar a lesão. Por isso, o recorte da vinheta e a segmentação fazem "
            "parte do pré-processamento, e o desempenho também será medido separadamente nas "
            "imagens sem vinheta.",
        ])
        + p("A base não fornece máscaras de segmentação; para avaliar a segmentação, a equipe "
            "anotará manualmente a região da lesão em um subconjunto de imagens do conjunto de teste.")
        + subtitulo("Questão científica")
        + p("A investigação será guiada pelas seguintes perguntas:")
        + topicos([
            "Uma abordagem tradicional e interpretável, baseada em características ABCD, de cor e de "
            "textura, consegue desempenho próximo ao de uma rede neural convolucional na triagem "
            "benigno × maligno?",
            "Qual é a contribuição de cada etapa de pré-processamento (remoção de pelos, normalização "
            "de cor, realce de contraste) para a qualidade da segmentação e da classificação?",
            "Os modelos baseiam suas decisões na lesão ou em artefatos da imagem, como pelos, réguas "
            "e bordas escuras do dermatoscópio?",
        ])
        + subtitulo("Métricas de avaliação")
        + tabela(["Métrica", "O que mede", "Por que é adequada"], [
            ("Sensibilidade (recall da classe maligna)",
             "Proporção de lesões malignas corretamente identificadas.",
             "Principal métrica em triagem: deixar de encaminhar um câncer é o erro mais grave."),
            ("Especificidade",
             "Proporção de lesões benignas corretamente identificadas.",
             "Mede a capacidade de evitar encaminhamentos desnecessários e sobrecarga das filas."),
            ("Precisão e F1-score",
             "Confiabilidade dos alertas e equilíbrio entre precisão e sensibilidade.",
             "Indicam quantos dos alertas emitidos são de fato melanomas, complementando a sensibilidade."),
            ("AUC-ROC e acurácia balanceada",
             "Desempenho em todos os limiares de decisão; média entre sensibilidade e especificidade.",
             "Permitem comparar os modelos independentemente do limiar e da proporção das classes."),
            ("Índice de Dice e IoU (Jaccard)",
             "Sobreposição entre a máscara segmentada e a máscara anotada manualmente.",
             "Avaliam objetivamente a qualidade da segmentação, da qual dependem as características."),
            ("Matriz de confusão",
             "Distribuição de acertos e erros por classe.",
             "Facilita a análise dos tipos de erro e a escolha do limiar de decisão."),
        ], [27, 33, 40])
        + p("Os modelos serão ajustados com validação cruzada estratificada no conjunto de treino "
            "e comparados no mesmo conjunto de teste, após a remoção das duplicatas. Além das métricas, serão registrados o tempo de "
            "processamento por imagem e uma análise qualitativa dos erros e dos mapas Grad-CAM.")
    )


def secao_referencias():
    refs = [
        "BRASIL. Instituto Nacional de Câncer (INCA). **Estimativa 2026–2028: incidência de câncer no "
        "Brasil** — estimativas por estado e capital. Rio de Janeiro: INCA, 2026. Disponível em: "
        "https://www.gov.br/inca/pt-br/assuntos/cancer/numeros/estimativa/estado-capital. "
        "Acesso em: 22 out. 2026.",
        "ESTEVA, A. et al. Dermatologist-level classification of skin cancer with deep neural "
        "networks. **Nature**, v. 542, p. 115–118, 2017.",
        "FINLAYSON, G. D.; TREZZI, E. Shades of gray and colour constancy. In: **Color and Imaging "
        "Conference**, 12., 2004, Scottsdale. Proceedings... IS&T, 2004. p. 37–41.",
        "JAVID, M. H. **Melanoma Skin Cancer Dataset of 10000 Images.** Kaggle, 2022. Licença CC0. "
        "Disponível em: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-"
        "10000-images. Acesso em: 22 out. 2026.",
        "JORNAL OPÇÃO. Em 10 anos, câncer de pele causou mais de 30 mil mortes no Brasil. Goiânia, "
        "jan. 2020. Com dados da Sociedade Brasileira de Dermatologia (SBD). Disponível em: "
        "https://www.jornalopcao.com.br/ultimas-noticias/em-10-anos-cancer-de-pele-causou-mais-de-30-"
        "mil-mortes-no-brasil-228656/. Acesso em: 22 out. 2026.",
        "LEE, T. et al. DullRazor: a software approach to hair removal from images. **Computers in "
        "Biology and Medicine**, v. 27, n. 6, p. 533–543, 1997.",
        "NACHBAR, F. et al. The ABCD rule of dermatoscopy. **Journal of the American Academy of "
        "Dermatology**, v. 30, n. 4, p. 551–559, 1994.",
        "O POVO. Ceará registra mais de 10 mil diagnósticos de câncer de pele não melanoma entre 2023 "
        "e 2025. Fortaleza, 11 jan. 2026. Disponível em: https://mais.opovo.com.br/jornal/ciencia-e-saude/"
        "2026/01/11/ceara-registra-mais-de-10-mil-diagnostico-de-cancer-de-pele-nao-melanoma-entre-"
        "2023-e-2025.html.",
        "SOCIEDADE BRASILEIRA DE DERMATOLOGIA (SBD). Diagnósticos de câncer de pele disparam no Brasil "
        "e expõem desigualdades regionais de acesso à saúde. Agora RN, 26 jan. 2026. Disponível em: "
        "https://agorarn.com.br/ultimas/diagnosticos-de-cancer-de-pele-disparam-no-brasil-e-expoem-"
        "desigualdades-regionais-de-acesso-a-saude/.",
        "YARAK, S.; RUIZ, E. E. S.; PISA, I. T. A teledermatologia na prática médica. **Revista "
        "Brasileira de Educação Médica**, v. 41, n. 2, 2017. DOI: 10.1590/1981-52712015v41n2RB20160042.",
    ]
    return secao("Referências") + '<div class="refs">' + "".join(p(r) for r in refs) + "</div>"


CSS = """
@page {
  size: A4;
  margin: 2cm 2.2cm 2.2cm;
  @bottom-center { content: counter(page); font: 10pt "Times New Roman", serif; color: #555; }
}
* { box-sizing: border-box; }
body { font: 11.5pt/1.45 "Times New Roman", Times, serif; color: #111; margin: 0; }
.faixa { background: #1a3dd6; color: #fff; text-align: center; font: bold 13pt Arial, sans-serif;
         padding: 10px; letter-spacing: .5px; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
table.cabecalho { width: 100%; border-collapse: collapse; margin: 14px 0 18px;
                  border-top: 1.5px solid #1a3dd6; border-bottom: 1.5px solid #1a3dd6; font-size: 10.5pt; }
table.cabecalho th, table.cabecalho td { text-align: left; vertical-align: top; padding: 5px 6px; }
table.cabecalho th { white-space: nowrap; }
h1 { font-size: 15pt; text-align: center; line-height: 1.3; margin: 8px 0 4px; }
.local { text-align: center; font-style: italic; margin: 0 0 18px; }
h2.secao { font-size: 12pt; text-transform: uppercase; text-align: center; background: #f2f2f2;
           border-top: 1.5px solid #000; border-bottom: 1.5px solid #000; padding: 6px; margin: 22px 0 12px;
           -webkit-print-color-adjust: exact; print-color-adjust: exact; break-after: avoid; }
h3 { font-size: 12pt; color: #1a3dd6; margin: 14px 0 6px; break-after: avoid; }
p { text-align: justify; margin: 0 0 8px; }
ul, ol { margin: 0 0 8px; padding-left: 22px; }
li { text-align: justify; margin-bottom: 5px; }
table.dados { width: 100%; border-collapse: collapse; table-layout: fixed; margin: 6px 0 12px; font-size: 10pt; }
table.dados th { background: #dce3fa; text-align: left; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
table.dados th, table.dados td { border: 1px solid #9aa6c8; padding: 5px 6px; vertical-align: top; line-height: 1.3; }
table.dados tr { break-inside: avoid; }
mark { background: #fff36b; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.refs p { text-align: left; margin-bottom: 7px; word-break: break-word; }
"""


def main():
    corpo = cabecalho() + secao_equipe() + secao_publico() + secao_proposta() + secao_referencias()
    SAIDA_HTML.write_text(
        f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        f"<title>PDI - Etapa 1 - Equipe e Proposta</title><style>{CSS}</style></head>"
        f"<body>{corpo}</body></html>",
        encoding="utf-8",
    )
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={SAIDA_PDF}", SAIDA_HTML.resolve().as_uri()],
        check=True, capture_output=True,
    )
    print(f"Gerado: {SAIDA_PDF}")


if __name__ == "__main__":
    main()
