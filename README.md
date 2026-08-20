# Projeto Integrador III — Grupo 3

## Explicabilidade e Visualização em Imagens de Câncer de Pulmão

Projeto desenvolvido no **Projeto Integrador III**, com foco na utilização de técnicas de **Explainable Artificial Intelligence (XAI)** para analisar e visualizar as decisões de modelos de Inteligência Artificial aplicados à detecção/classificação de nódulos pulmonares em imagens de tomografia computadorizada (TC).

---

## 📌 Resumo

O projeto busca investigar como modelos de Inteligência Artificial podem auxiliar na análise de imagens de TC para detecção de nódulos pulmonares e, principalmente, **como explicar as decisões tomadas por esses modelos**.

O **Grupo 3** é responsável pela parte de **Explicabilidade e Visualização**, tendo como objetivo investigar se as regiões destacadas pelos modelos realmente correspondem às regiões relevantes dos nódulos pulmonares.

A abordagem definida inicialmente envolve o uso do dataset **LIDC-IDRI**, modelos de classificação e técnicas de XAI, tendo o **Grad-CAM** como principal método de referência. Posteriormente, será realizada uma comparação com pelo menos uma técnica adicional, como **SHAP, Integrated Gradients ou LIME**.

A avaliação deverá considerar não apenas a visualização dos mapas de calor, mas também métricas como **IoU**, teste de sanidade de **Adebayo** e **Expected Calibration Error (ECE)**.

---

## 🎯 Objetivo

Investigar técnicas de explicabilidade aplicadas a modelos de classificação de nódulos pulmonares, buscando compreender:

> **Por que o modelo tomou determinada decisão e se as regiões destacadas por ele correspondem às regiões relevantes do nódulo.**

O projeto não busca apenas verificar se o modelo acerta, mas também analisar **como e onde o modelo está tomando suas decisões**.

---

## 🧪 Estado atual do projeto

**Status: 🟡 Em desenvolvimento / Fase de estudo**

Neste momento, o grupo está concentrado na **fundamentação teórica, organização do projeto e preparação dos dados e ferramentas**.

---

## 🗂️ Dataset

O dataset principal definido para o projeto é o **LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative)**.

Ele possui exames de tomografia computadorizada de tórax com anotações realizadas por radiologistas e utiliza o formato **DICOM**. O dataset será utilizado como base para os experimentos de classificação e explicabilidade.

> ⚠️ Os arquivos do dataset não serão armazenados diretamente neste repositório devido ao seu grande tamanho.

---

## 🔬 Abordagem planejada

A ideia inicial do projeto pode ser resumida no seguinte fluxo:

```text
LIDC-IDRI
    ↓
Pré-processamento das imagens
    ↓
Modelo de classificação
    ↓
Predição
    ↓
Técnicas de XAI
    ↓
Mapas de calor / ativação
    ↓
Comparação com máscaras dos nódulos
    ↓
Avaliação das explicações
```

O **Grad-CAM** será utilizado como baseline, enquanto pelo menos um método adicional deverá ser investigado para comparação.

---

## 📊 Métricas planejadas

As principais métricas e avaliações definidas inicialmente são:

| Avaliação                        | Objetivo                                                                |
| -------------------------------- | ----------------------------------------------------------------------- |
| **IoU**                          | Verificar a sobreposição entre o mapa de ativação e a máscara do nódulo |
| **Teste de Sanidade de Adebayo** | Verificar se a explicação realmente depende do modelo                   |
| **ECE**                          | Avaliar a calibração das probabilidades do modelo                       |
| **Comparação entre métodos**     | Verificar diferenças entre as técnicas de XAI                           |

A documentação do projeto estabelece **IoU ≥ 0,40** como meta inicial e considera IoU e o teste de sanidade como avaliações prioritárias.

---

## 🛠️ Tecnologias em estudo

As principais tecnologias e bibliotecas consideradas neste momento são:

* **Python**
* **PyTorch**
* **Torchvision**
* **NumPy**
* **Matplotlib**
* **scikit-learn**
* **pydicom**
* **SimpleITK**
* **pylidc**
* **3D Slicer**

Essas ferramentas serão utilizadas conforme as necessidades das próximas etapas do projeto.

---

## 📚 Metodologia

O projeto segue o framework **KDD (Knowledge Discovery in Databases)**, passando pelas etapas de:

**Seleção → Pré-processamento → Transformação → Mineração → Interpretação**

No momento, o grupo encontra-se principalmente nas etapas iniciais de **seleção, preparação e estudo dos dados e métodos**.

---

## ⚠️ Observação

Este é um **projeto acadêmico em desenvolvimento**. Os métodos, ferramentas e estrutura apresentados neste README representam o estado atual do planejamento e poderão ser modificados conforme os experimentos e estudos avancem.

Os resultados deste projeto não devem ser utilizados para diagnóstico ou tomada de decisões médicas reais.

---

**Tema:** Processamento de Imagens de Câncer de Pulmão
**Foco:** Explainable AI (XAI) e visualização de decisões de modelos de IA
