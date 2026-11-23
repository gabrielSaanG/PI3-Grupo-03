# Projeto Integrador III — Grupo 3

## Explicabilidade em nódulos pulmonares (LIDC-IDRI)

**Pergunta que o projeto responde:**

> As regiões que o modelo destaca para classificar um nódulo pulmonar correspondem ao nódulo anotado pelos radiologistas?

Entrada única da entrega: [`notebooks/pipeline_entrega.ipynb`](notebooks/pipeline_entrega.ipynb).

---

## Como rodar

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. Coloque o checkpoint em `models/resnet50_v2/best_model.pth` (pesos não entram no Git).
2. Garanta DICOM local em `data/LIDC-IDRI/` e os CSVs já gerados em `outputs/sprint5/` (ou regenere com os scripts do pipeline).
3. Abra e execute **Run All** em `notebooks/pipeline_entrega.ipynb`.

O notebook **não retreina**. Ele carrega a ResNet50 v2, aplica o tratamento de dados do pipeline v2, gera Grad-CAM e Integrated Gradients em uma amostra do teste e, no final, mostra tabelas, gráficos e overlays com a ROI do radiologista.

---

## Pipeline v2 (resumo)

| Etapa | Decisão |
|------|---------|
| Dados | LIDC-IDRI, split por paciente **559** (391 / 84 / 84), seed 42 |
| Rótulo | consenso por nódulo físico; maligno se média ≥ 3,5; ambíguos (2,5–3,5) fora do treino/val |
| Crop | **64 mm** centrados na ROI → **224×224**, HU [-1000, 400] |
| Modelo | ResNet50 (ImageNet) → 2 classes — `models/resnet50_v2/` |
| XAI | Grad-CAM (`layer4`) + Integrated Gradients (+ baselines aleatório e centrado) |
| Métricas | IoU / Dice (limiar da validação), Pointing Game (±4,3 mm), distância do pico |

Figuras e índice da entrega: [`docs/entrega/`](docs/entrega/). Scripts de reprodução: `scripts/sprint5_*.py`, `scripts/sprint6_*.py`.

---

## Estrutura

```text
notebooks/pipeline_entrega.ipynb   ← entrega
models/resnet50_v2/                ← config + histórico (+ checkpoint local)
src/xai/                           ← Grad-CAM reutilizável
scripts/                           ← reprodução do pipeline / XAI
docs/entrega/                      ← figuras e índice da entrega final
auditoria_dataset/                 ← auditoria do split/rótulo
archive/                           ← material legado (inclui notas antigas por sprint)
data/                              ← LIDC local (gitignored)
outputs/                           ← manifests, métricas, cache (gitignored)
```

Material antigo (CNN custom, fusion, stubs, notas de sprint, etc.) está em [`archive/`](archive/README.md).

---

## Resultado (leitura curta)

Com crops centrados no nódulo, IoU alto **sozinho** não prova localização: o mapa centrado (sem aprendizado) passa de IoU 0,40. No teste do v2 (provisório), Grad-CAM e IG ficam abaixo desse baseline; o Grad-CAM frequentemente destaca pleura/parede. A pergunta do projeto exige comparar com o centrado e com os testes de deslocamento / Adebayo.

Figuras: [`docs/entrega/figures/`](docs/entrega/figures/).

---

## Observação

Projeto acadêmico. Resultados **provisórios** (uma semente, sem ensemble). Não usar para diagnóstico clínico.

**Tema:** Processamento de imagens de câncer de pulmão · **Foco:** XAI (Grad-CAM + IG)
