# Sprint 6 — Conferência dos números antes da Review 6 (Gabriel)

Objetivo: antes de enviar a Review 6, garantir que slides e documentos usem **a mesma acurácia, o mesmo conjunto de dados e a mesma regra de rótulo** em todo lugar.

**Escopo do que foi conferido aqui.** A tabela de registro do Wesley, os slides e as respostas de D1 a D4 ficam na página da sprint, fora do repositório. Por isso, esta conferência faz duas coisas:

1. lista, com o arquivo de origem, **todos os números do repositório** que podem acabar num slide, e marca os que conflitam entre si;
2. deixa uma **lista de checagem** para comparar slide por slide com a tabela de registro.

O que ainda falta é a comparação final com a tabela do Wesley e com as decisões D1 a D4 registradas na página. Ela está marcada como `[ ]` no fim.

---

## 1. Conjunto de dados: qual base é a oficial

| Número | O que é | Onde aparece | Situação |
|---|---|---|---|
| **559 pacientes (391/84/84), seed 42** | Divisão auditada por paciente. São todos os pacientes elegíveis (nódulo ≥ 3 mm com nota) que existem nesta máquina | `outputs/sprint3/split_summary.csv`, `auditoria_dataset/LEIAME.md`, `docs/sprint5/*` | Base de todos os números da Sprint 5 e da Sprint 6 |
| 18.508 / 3.694 / 4.163 patches | Patches por conjunto na divisão de 559 | `split_summary.csv`, `split_manifest.csv` | Consistente |
| 757 pacientes | Base do notebook do André (SEG/SR do Kaggle, validação cruzada, rótulo ≥ 3,5) | `docs/sprint5/resumo_sprint_review_gabriel.md` (slide 8) | **Outra base.** Não pode ser misturada com os números da divisão de 559 |
| 1.010 / 709 / 642 / 627 | Pastas de paciente / com DICOM / com TC / com anotação ligada a uma TC | `auditoria_dataset/LEIAME.md` | Só descrevem o disco. Não são o tamanho do conjunto usado |
| 12 pacientes, 27 nódulos (3 no teste) | Rodada piloto de 14/09 | `outputs/paper/table_1_dataset_split.csv` | **Obsoleto.** Não usar |
| 90 casos de teste | Teste dos modelos antigos (CNN, MLP semântico, fusão) | `models/fusion_cnn_mlp/final_model_comparison_table.csv` | **Outro teste.** Não comparável |
| 165 imagens, 3 classes | IQ-OTH/NCCD (Kaggle), usado só para validar o pipeline | `docs/anotations/Anotações-P5/...md` | **Outro dataset.** Não é LIDC |

**Unidade de contagem no teste (mesma divisão):** são 4.163 patches, 687 nódulos de leitor (paciente, exame, leitor e `nodule_id`) e 275 nódulos físicos. O `split_summary.csv` diz 454 "nódulos" porque conta `nodule_id` distintos sem considerar o paciente. **Esse 454 não deve aparecer em slide.**

## 2. Regra de rótulo: há três no repositório

| Regra | Onde | Prevalência no teste |
|---|---|---|
| **v1:** cada leitor, nota ≥ 3 = maligno (o 3, "indeterminado", conta como maligno) | `notebooks/03-resnet50-sprint3.ipynb` (`patches_df["malignancy"] >= 3`) | 75,4% dos patches e 70,2% dos 687 nódulos |
| **v2 (consenso):** média dos leitores do mesmo nódulo físico ≥ 3,5 = maligno; ambíguos (2,5 < média < 3,5) fora do treino e da validação | `scripts/sprint5_retrain_resnet50.py`, `auditoria_dataset` (`LIMIAR_MALIGNO = 3.5`) | 41,4% dos patches e 26,1% dos 687 nódulos; sem os ambíguos, 47,1% de 380 nódulos (72 pacientes) |
| Teste de vazamento D5: nota ≥ 4 = maligno, com o 3 excluído ou contado como benigno | `auditoria_dataset/Auditoria_Dataset_Oficial.ipynb`, seção 7 | Só para o teste da ficha. Não é rótulo de modelo |

- A frase **"cerca de 75% maligno"** só vale para a regra v1. Na regra v2 ela fica errada.
- O rótulo oficial depende da decisão do grupo (D6 na Sprint 5). Ele tem que ser **um só** em todos os slides, e o slide precisa dizer qual é.

## 3. Acurácia: há pelo menos oito números diferentes

Todos vêm da mesma divisão de 559 pacientes, mas mudam conforme o rótulo, a unidade e o limiar:

| Modelo | Rótulo | Unidade | Acurácia | Referência "sempre maligno" | Fonte |
|---|---|---|---|---|---|
| ResNet50 v1 | v1 | patch | **0,699** (os "70%" antigos) | 0,754 | `outputs/sprint3/test_results_resnet50.csv` |
| ResNet50 v1 | v1 | nódulo (687) | 0,674 | 0,702 | `outputs/sprint5/metrics_provisorio.csv` |
| ResNet50 v1 | v2 | nódulo (687) | 0,582 | 0,261 | `metrics_provisorio_v2_todos.csv` |
| ResNet50 v2 | v2 | patch | 0,773 | 0,414 | `metrics_patch_reference_provisorio_v2_todos.csv` |
| ResNet50 v2 | v2 | nódulo (687), limiar 0,5 | **0,789** [0,720; 0,853] | 0,261 | `metrics_provisorio_v2_todos.csv` |
| ResNet50 v2 | v2 | nódulo (687), limiar da validação | 0,808 | 0,261 | idem |
| ResNet50 v2 | v2 | nódulo sem ambíguos (380), limiar 0,5 | 0,882 | 0,471 | `metrics_provisorio_v2_sem_ambiguos.csv` |
| Fusão CNN-MLP (antiga) | outro | 90 casos | 0,944 | — | `models/fusion_cnn_mlp/final_model_comparison_table.csv` |
| ResNet50 no IQ-OTH/NCCD | 3 classes | imagem | 0,97 | — | `docs/anotations/Anotações-P5` |

**Regras para os slides:**

1. A métrica principal é a **AUC** e a **acurácia balanceada**, por nódulo e com IC 95% por paciente. Se a acurácia aparecer, deve ser **uma só** (a registrada na tabela do Wesley) e vir sempre **ao lado do "sempre maligno"** com o mesmo rótulo.
2. Os 97% (IQ-OTH/NCCD), os 94% (fusão, que usa a ficha do radiologista; ver D5) e as tabelas de `outputs/paper/` (teste com 3 nódulos, acurácia 0 ou 1 e AUC indefinida) **não podem aparecer como resultado do projeto**.
3. Quando a frase disser "70%", conferir se é o 0,699 por patch do v1 e se o slide diz isso. Esse número não é comparável com nada do v2.

## 4. Outros pontos que conflitam

- **`README.md` está desatualizado.** Fala em "crops LIDC 64×64", põe a `custom_cnn` como fluxo principal e diz que o grupo está "nas etapas iniciais". Também trata IoU ≥ 0,40 como meta, mas o mapa centrado (que não aprende nada) passa dela (IoU de 0,50). Antes da Review 6, atualizar ou deixar claro que é o planejamento inicial.
- **Teste de Adebayo antigo** (`outputs/sprint3/adebayo_*`): é da rodada pequena, com teste de 598 patches, e não da divisão de 559. O teste no v2 foi refeito nesta sprint (`docs/sprint6/gabriel_sprint6.md`).
- **Contagem de anotações na auditoria.** O `resumo_auditoria.md` diz 7.034 anotações (todas, uma linha por radiologista). O `LEIAME.md` diz 4.552 (só as com nota, de 559 pacientes). Os dois estão certos, mas medem coisas diferentes. Se aparecer em slide, dizer qual é.
- **Status.** Todos os números do v1 e do v2 são **provisórios**: uma semente, sem ensemble e não são o modelo oficial do André. Todo slide com esses números precisa da palavra "provisório".

## 5. Lista de checagem para a Review 6 (slide por slide)

- [ ] O conjunto de dados está escrito como "559 pacientes (391/84/84), divisão por paciente, seed 42", e nenhum slide usa 757, 90 casos, 27 nódulos ou o IQ-OTH/NCCD como resultado.
- [ ] A regra de rótulo é a mesma em todos os slides e é a registrada na tabela (v1 ≥ 3 por leitor **ou** v2 média ≥ 3,5), com a prevalência correspondente.
- [ ] A acurácia é a mesma em todos os slides e é a registrada, com a unidade (patch ou nódulo), o limiar e o "sempre maligno" ao lado.
- [ ] AUC e acurácia balanceada aparecem com IC 95% por paciente.
- [ ] Visão com ambíguos e sem ambíguos lado a lado (se o rótulo for o v2).
- [ ] MLP semântico e fusão estão marcados como "não comparáveis" (D5).
- [ ] Números do v2 com o selo "provisório".
- [ ] A explicabilidade cita o mapa centrado como referência (sem ele, IoU e Pointing Game não provam nada nestes recortes).
- [ ] **Comparar cada número dos slides com a tabela de registro do Wesley e com as respostas de D1 a D4 da página** (não está no repositório; precisa ser feito com a tabela aberta).
