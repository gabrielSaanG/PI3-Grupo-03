# Sprint 6 — Sanidade da explicabilidade do v2, pleura e conferência dos números (Gabriel)

**Status de todos os números abaixo: PROVISÓRIO.** Eles vêm do ResNet50 v2 de semente única (`models/resnet50_v2/best_model.pth`), na divisão auditada de 559 pacientes (391/84/84, seed 42) e com o rótulo de consenso (média dos leitores ≥ 3,5). Valem como resultado comparativo só se o grupo adotar o v2 e esse rótulo (decisão da página da sprint).

## Atividades da sprint

| Atividade | Situação |
|---|---|
| Reunião de decisões D1 a D4 (04/11) | Feita (respostas registradas na página da sprint, fora do repositório) |
| Métricas e IC com as previsões oficiais, ao lado do "sempre maligno" | Marcada como feita. Os arquivos da rodada oficial (`metrics_oficial*.csv`) **não estão nesta máquina**; se foi rodada em outra, copiar para `outputs/sprint5/` |
| Conferir slides e documentos contra a tabela de registro | Conferência do repositório feita em [`conferencia_registro.md`](conferencia_registro.md). Falta comparar com a tabela do Wesley e com os slides, que não estão no repositório |
| Adebayo e teste de deslocamento no v2 | **Feito** (seções 1 e 2) |
| Pleura e parede torácica como atalho de contexto | **Feito** (seção 3) |

## Amostra e protocolo comuns

- **Mesma amostra da Sprint 5:** um patch por nódulo físico (o de maior ROI), 275 nódulos de 84 pacientes no teste. O script confere caso a caso contra `outputs/sprint5/xai_v2_per_case.csv` (275 de 275 iguais).
- **Mesmos mapas e limiares:** Grad-CAM na `layer4` e Integrated Gradients (baseline de ar, 32 passos), para a classe prevista. Os limiares de IoU são os escolhidos na validação na Sprint 5 (Grad-CAM 0,5, IG 0,1, centrado 0,9). Pointing Game: o pico cai na ROI dilatada em 15 px (cerca de 4,3 mm).
- **IC 95%:** bootstrap por paciente, 1.000 reamostragens.
- **AUC por nódulo físico.** Nesta amostra (147 nódulos não ambíguos, um patch cada), o v2 tem AUC 0,932. **Esse número não substitui o oficial** (0,949 em 380 nódulos de leitor, com a média dos patches). Serve só para comparar condições dentro de cada teste.
- O v2 refeito aqui reproduziu os números da Sprint 5: Pointing Game de 42,2% (Grad-CAM) e 66,5% (IG). As probabilidades diferem em até 0,035 das salvas na Sprint 5, porque lá a inferência foi em meia precisão (fp16) e aqui em fp32.

---

## 1. Teste de sanidade de Adebayo no v2

Script: `scripts/sprint6_v2_adebayo.py` (cerca de 25 minutos). Saídas: `outputs/sprint6/adebayo_v2_per_case.csv`, `adebayo_v2_summary.csv`, `preds_*_resnet50_v2_labelrand.csv` e `docs/sprint6/figures/adebayo_v2_*.png`.

### 1.1 Randomização dos pesos em cascata

**O que foi feito:** a partir do v2 treinado, os blocos são reinicializados de cima para baixo (`fc`, depois `layer4`, `layer3`, `layer2`, `layer1` e `conv1`), acumulando. Em cada etapa, os mapas são recalculados para a classe prevista pelo v2 original e comparados com os mapas do v2 (correlação de Spearman) e com a ROI do leitor.

**O que se espera:** se o mapa depende do que o modelo aprendeu, ele muda logo nas primeiras etapas (Spearman perto de 0) e deixa de cair no nódulo.

| Etapa (blocos aleatórios) | AUC (147 não ambíguos) | Grad-CAM: Spearman com o v2 (mediana) | Grad-CAM: Pointing Game | IG: Spearman com o v2 (mediana) | IG: Pointing Game |
|---|---|---|---|---|---|
| v2 treinado | 0,932 [0,880; 0,970] | 1 | 42,2% [36,9; 48,3] | 1 | 66,5% [58,6; 74,8] |
| até `fc` | 0,138 | −0,02 | 9,5% | 0,77 | 44,7% |
| até `layer4` | 0,410 | 0,06 | 4,4% | 0,67 | 4,0% |
| até `layer3` | 0,472 | 0,03 | 1,1% | 0,62 | 2,5% |
| até `layer2` | 0,533 | 0,01 | 2,2% | 0,56 | 2,5% |
| até `layer1` | 0,573 | 0,15 | 4,0% | 0,48 | 3,3% |
| rede inteira (até `conv1`) | 0,478 | −0,04 | 6,5% | 0,43 | 4,0% |

![Adebayo no v2: Spearman e Pointing Game por etapa](figures/adebayo_v2_curvas.png)

**Leitura:**
- **O Grad-CAM passa.** Com só o `fc` aleatório, o mapa já não tem relação com o original (Spearman −0,02) e o Pointing Game cai de 42% para 9,5%.
- **O IG passa no Pointing Game, mas não na correlação.** O pico do IG deixa de cair no nódulo quando a `layer4` é aleatorizada (de 66,5% para 4%). Isso mostra que **acertar o nódulo depende dos pesos treinados**. Mas o mapa inteiro continua parecido com o original mesmo com a rede toda aleatória (Spearman 0,43), e a correlação do IG com a própria imagem é 0,65 no v2 e 0,48 na rede aleatória. Isso acontece porque o IG multiplica o gradiente pela diferença entre a imagem e o baseline, e o resultado acompanha o brilho da imagem. **Conclusão:** para o IG, vale citar o pico (Pointing Game), mas não a semelhança visual do mapa.
- O AUC de 0,14 com só o `fc` aleatório indica que as features da `layer4` ainda separam as classes; o `fc` aleatório apenas inverte o sinal. A partir da `layer4`, o modelo fica no acaso, como esperado.
- Na rede aleatória, de 13% a 67% dos mapas do Grad-CAM ficam vazios (tudo zero depois do ReLU). Esses casos contam como erro no Pointing Game e ficam fora da mediana do Spearman.

### 1.2 Rótulos embaralhados (refeito na divisão de 559)

**O que foi feito:** os rótulos de consenso do treino e da validação foram permutados **entre nódulos físicos** (seed 42). Todas as fatias de um nódulo recebem o mesmo rótulo sorteado, e a proporção de classes é mantida. Um ResNet50 novo (ImageNet) foi treinado com a mesma receita do v2 por 5 épocas, sem escolher época pelos rótulos aleatórios. O desempenho foi medido no **teste completo, com os rótulos verdadeiros**, com o mesmo `scripts/sprint5_evaluate.py`.

Isso resolve as três limitações do teste da Sprint 3: agora é a divisão de 559, a permutação não é por patch (o modelo não consegue acertar decorando fatias vizinhas do mesmo nódulo) e o desempenho é medido no teste inteiro.

**Treino:** a acurácia de treino nos rótulos aleatórios subiu de 70% para 95% em 5 épocas, ou seja, o modelo **decorou** os rótulos. A AUC da validação contra os rótulos aleatórios ficou entre 0,45 e 0,50. Os rótulos sorteados coincidiram com os verdadeiros em 57,3% dos nódulos de treino; o esperado por acaso, com essa proporção de classes, é cerca de 55%.

**Teste completo (687 nódulos de leitor, 84 pacientes, limiar 0,5).** Arquivos: `outputs/sprint5/metrics_provisorio_v2_labelrand_todos.csv` e `..._sem_ambiguos.csv`.

| Métrica | Sempre maligno | v2 | v2 com rótulos embaralhados |
|---|---|---|---|
| AUC | 0,500 | 0,906 [0,852; 0,948] | **0,555 [0,464; 0,652]** |
| Acurácia balanceada | 0,500 | 0,812 [0,747; 0,866] | 0,562 [0,488; 0,655] |
| Acurácia | 0,261 | 0,789 | 0,623 |
| TN/FP/FN/TP | 0/508/0/179 | 388/120/25/154 | 350/158/101/78 |
| AUC sem ambíguos (380 nódulos, 72 pacientes) | 0,500 | 0,949 [0,901; 0,984] | **0,523 [0,416; 0,622]** |

**Mapas do modelo com rótulos embaralhados** (mesmos 275 nódulos):

| | Spearman com o v2 (mediana [IQR]) | Pointing Game |
|---|---|---|
| Grad-CAM | 0,28 [−0,01; 0,50] | 8,7% [5,5; 13,2] (v2: 42,2%) |
| IG | 0,65 [0,58; 0,73] | 36,7% [30,8; 43,8] (v2: 66,5%) |

**Leitura:**
- **O teste agora passa.** O modelo com rótulos aleatórios fica no acaso no teste (os dois ICs da AUC contêm 0,5), enquanto o v2 fica bem acima. O alerta da Sprint 3 (62% contra 51%) era um efeito da amostra pequena e da permutação por patch.
- **A acurácia volta a enganar:** o modelo aleatório tem 62,3% de acurácia, mais que o dobro do "sempre maligno" (26,1%), só porque responde "benigno" quase sempre, e 74% dos nódulos são benignos.
- **Grad-CAM:** os mapas mudam (Spearman 0,28) e quase nunca caem no nódulo (8,7%). Passa.
- **IG:** o mapa continua parecido (0,65, de novo pelo brilho da imagem), e o pico ainda cai no nódulo em 37% dos casos. Esse modelo foi treinado em recortes centrados no nódulo e precisou olhar para o nódulo para decorar os rótulos, então parte desse acerto é esperada. Mesmo assim, o v2 (66,5% [58,6; 74,8]) fica claramente acima, sem sobreposição dos ICs.

![Mapas do v2 e dos modelos aleatorizados](figures/adebayo_v2_exemplos.png)

---

## 2. Teste de deslocamento (nódulo fora do centro)

Script: `scripts/sprint6_v2_shift_pleura.py` (cerca de 15 minutos, junto com a seção 3). Saídas: `outputs/sprint6/deslocamento_v2_*.csv` e `docs/sprint6/figures/deslocamento_v2_*.png`.

**Por que:** na Sprint 5, o mapa centrado (uma gaussiana no centro, que não aprende nada) venceu todos os outros (IoU 0,50 e Pointing Game de 100%), porque todo recorte tem o nódulo no centro. Para saber se o Grad-CAM e o IG localizam o nódulo de verdade, o nódulo precisa sair do centro.

**O que foi feito:** o centro do recorte de 64 mm foi movido 8 e 16 mm para a direita, a esquerda, cima e baixo (4 direções por nódulo). A ROI acompanha o deslocamento. O v2 e os mapas foram recalculados em cada posição. No treino, o v2 só viu deslocamentos de até 12 px (cerca de 3,4 mm).

| Deslocamento | AUC do v2 (não ambíguos) | Classe trocada | Grad-CAM: Pointing Game | IG: Pointing Game | Centrado: Pointing Game |
|---|---|---|---|---|---|
| 0 mm | 0,932 [0,880; 0,970] | — | 42,2% [36,9; 48,3] | 66,5% [58,6; 74,8] | 100% |
| 8 mm | 0,922 [0,870; 0,962] | 8% | 38,0% [32,0; 44,9] | 59,5% [51,9; 68,3] | 43,3% [34,9; 52,2] |
| 16 mm | 0,860 [0,795; 0,913] | 16% | **26,5% [21,2; 32,8]** | **37,4% [31,4; 44,2]** | **2,7% [1,4; 4,4]** |

Pointing Game com 16 mm de deslocamento, por tamanho do nódulo: < 6 mm: Grad-CAM 14%, IG 17%; 6–10 mm: 19% e 34%; 10–20 mm: 47% e 65%; ≥ 20 mm: 74% e 74%.

![Pointing Game e AUC com o nódulo deslocado](figures/deslocamento_v2_curvas.png)

![Exemplos do teste de deslocamento](figures/deslocamento_v2_exemplos.png)

**Leitura:**
- **Com o nódulo fora do centro, o Grad-CAM e o IG vencem o mapa centrado.** A 16 mm, o centrado cai para 2,7%, e o Grad-CAM (26,5%) e o IG (37,4%) ficam muito acima, sem sobreposição dos ICs. Os mapas carregam informação real de localização, não só o viés do centro. Esta é a comparação que responde à crítica dos mentores.
- **Os mapas não ficam presos no centro.** A 16 mm, o pico fica a menos de 4,3 mm do centro do recorte em só 8,8% dos casos (Grad-CAM) e 3,3% (IG).
- **Mas a localização piora com o deslocamento.** Dos casos em que o mapa acertava o nódulo centrado, só 47% (Grad-CAM) e 52% (IG) continuam acertando a 16 mm. Nos nódulos menores que 10 mm, o acerto fica abaixo de 35%.
- **O modelo também depende do nódulo no centro.** A AUC cai de 0,932 para 0,860 a 16 mm, e 16% das previsões trocam de classe. Para o próximo treino, vale aumentar o deslocamento na augmentation.

---

## 3. Pleura e parede torácica como atalho de contexto

Mesmo script da seção 2. Saídas: `outputs/sprint6/pleura_v2_*.csv` e `docs/sprint6/figures/pleura_v2_exemplos.png`.

**Pergunta:** na Sprint 5, o Grad-CAM do v2 marcou com frequência a pleura ou a parede torácica. O modelo usa esse contexto como atalho para decidir?

**Protocolo:**
- **Pulmão:** segmentado na fatia inteira (ar abaixo de −400 HU sem tocar a borda da imagem, as duas maiores partes, fechamento de 5 mm e preenchimento de buracos). Cada recorte é dividido em **nódulo** (ROI dilatada em 4,3 mm, a tolerância do Pointing Game), **pulmão** e **fora do pulmão** (pleura, parede torácica e mediastino).
- **Justapleural:** a ROI fica a até 2 mm da região fora do pulmão (111 de 275 nódulos, 40%).
- **Oclusão:** o v2 é recalculado depois de preencher com tecido de pulmão (−850 HU): (a) tudo o que está fora do pulmão, (b) o nódulo (ROI dilatada em 2 mm), (c) tudo menos o nódulo.

### 3.1 Onde o pico cai

| Mapa | Pico no nódulo | Pico no pulmão | Pico fora do pulmão | Massa fora do pulmão / área fora do pulmão |
|---|---|---|---|---|
| Grad-CAM | 42,2% | 29,5% | **28,4% [22,0; 34,1]** | 0,75 |
| IG | 66,5% | 2,5% | 30,9% | 1,48 |
| Aleatório | 2,9% | 47,6% | 48,7% | 1,00 |
| Centrado | 100% | 0% | 0% | 0,38 |

A região fora do pulmão ocupa, em média, 39% do recorte; o nódulo (com a margem), 7%.

**Quando o Grad-CAM cai fora do pulmão:**

| Grupo | n | Pico do Grad-CAM fora do pulmão |
|---|---|---|
| Nódulo justapleural (≤ 2 mm) | 111 | **45,0%** |
| Nódulo longe da pleura | 164 | 17,1% |
| v2 previu benigno | 182 | **37,4%** |
| v2 previu maligno | 93 | 10,8% |
| Acerto do v2 / erro do v2 / ambíguo | 124 / 23 / 128 | 19,4% / 30,4% / 36,7% |

### 3.2 O contexto ajuda a prever o rótulo?

| Preditor, sem olhar o nódulo (147 não ambíguos) | AUC para "maligno" |
|---|---|
| Proximidade da pleura | 0,477 [0,365; 0,595] |
| Justapleural (sim ou não) | 0,487 [0,389; 0,593] |
| Área fora do pulmão no recorte | 0,347 [0,245; 0,462] |

Entre os não ambíguos, 34,5% dos justapleurais são malignos, contra 37,1% dos que ficam longe da pleura. Os recortes de benignos têm mais área fora do pulmão (41% contra 33%), por isso a AUC abaixo de 0,5 na última linha.

### 3.3 Oclusão

| Condição (147 não ambíguos, 72 pacientes) | AUC | p médio dos malignos | p médio dos benignos | Classe trocada |
|---|---|---|---|---|
| Original | 0,932 [0,880; 0,970] | 0,788 | 0,179 | — |
| Sem o que está fora do pulmão | **0,933 [0,890; 0,969]** | 0,837 | 0,266 | 10,9% |
| Só o nódulo (contexto todo removido) | **0,929 [0,886; 0,966]** | 0,843 | 0,290 | 15,6% |
| Sem o nódulo | **0,756 [0,671; 0,831]** | 0,548 | 0,296 | 21,1% |

![Pleura e parede torácica: exemplos e oclusão](figures/pleura_v2_exemplos.png)

**Leitura:**
- **A parede torácica não é um atalho para separar as classes.** Ao apagar tudo o que está fora do pulmão, a AUC fica igual (0,933 contra 0,932). Ao deixar só o nódulo, também (0,929). Além disso, estar perto da pleura não prevê o rótulo neste teste (AUC 0,48).
- **A separação vem do nódulo.** Sem o nódulo, a AUC cai para 0,756 e o p médio dos malignos cai de 0,79 para 0,55. A AUC continua acima de 0,5, o que pode vir das bordas do nódulo fora da ROI (espículas), ou do próprio buraco deixado pela oclusão, cujo tamanho revela o tamanho do nódulo. Este teste não separa as duas coisas.
- **O que o Grad-CAM mostra na parede é, sobretudo, a explicação da classe "benigno".** O pico cai fora do pulmão em 37% das previsões de benigno e em 11% das de maligno, e quase sempre em nódulos colados à pleura (45%). Em média, o Grad-CAM **não** se concentra fora do pulmão (massa/área de 0,75, abaixo de 1), mas se concentra no nódulo (4,6 vezes acima da área).
- **O contexto mexe na probabilidade, não na ordenação.** Sem a região fora do pulmão, as probabilidades sobem (benignos de 0,18 para 0,27) e 11% das previsões trocam de classe. Ou seja, o contexto empurra para "benigno" e afeta o limiar e a calibração. Como os benignos têm mais área fora do pulmão no recorte, isso é um **risco de atalho** em outra base, a ser registrado, mas não aparece como atalho no teste atual.
- **O IG se concentra fora do pulmão** (massa/área de 1,48). Isso combina com a seção 1: o IG acompanha o brilho da imagem, e a parede torácica é clara.

**Limitações da seção 3:**
- a oclusão cria imagens que o modelo nunca viu (bordas artificiais e um valor constante);
- a segmentação do pulmão é heurística e não foi validada caso a caso; ela falha em alguns cortes (por exemplo, a última linha da figura);
- "fora do pulmão" junta pleura, parede torácica e mediastino;
- são 147 nódulos não ambíguos, um patch por nódulo, de um modelo de uma semente.

---

## 4. Conferência dos números para a Review 6

Ver [`conferencia_registro.md`](conferencia_registro.md). Em resumo, o repositório tem:
- **três bases** que podem ser confundidas (559 pacientes auditados, 757 do André, e as rodadas antigas de 27 nódulos e de 90 casos), além do IQ-OTH/NCCD (outro dataset, 97% de acurácia);
- **três regras de rótulo** (v1 ≥ 3 por leitor; v2 média ≥ 3,5; ≥ 4 no teste de vazamento D5);
- **pelo menos oito acurácias diferentes** na mesma divisão: de 0,58 a 0,88, conforme o rótulo, a unidade e o limiar.

Falta comparar slide por slide com a tabela do Wesley e com as respostas de D1 a D4, que estão na página da sprint.

## 5. Como reproduzir

```text
.venv\Scripts\python.exe scripts\sprint6_v2_adebayo.py
.venv\Scripts\python.exe scripts\sprint6_v2_shift_pleura.py
.venv\Scripts\python.exe scripts\sprint5_evaluate.py --pred resnet50_v2=outputs/sprint5/preds_test_resnet50_v2.csv --pred v2_rotulos_aleatorios=outputs/sprint6/preds_test_resnet50_v2_labelrand.csv --val resnet50_v2=outputs/sprint5/preds_val_resnet50_v2.csv --val v2_rotulos_aleatorios=outputs/sprint6/preds_val_resnet50_v2_labelrand.csv --manifest outputs/sprint5/split_manifest_v2.csv --status provisorio --tag v2_labelrand_todos
```

Para a visão sem ambíguos, rode o último comando com `--exclude-ambiguous --tag v2_labelrand_sem_ambiguos`. Os dois primeiros precisam de `D:\datasetLIDC_completo` e de GPU, e os logs ficam em `outputs/sprint6/*_log.txt`. O modelo com rótulos embaralhados fica em `models/resnet50_v2_labelrand/`.

O código comum (divisão, recorte de 64 mm, ROI, segmentação do pulmão, mapas e bootstrap) está em `scripts/sprint6_v2_common.py`.

## Pendências

- [ ] Comparar os slides com a tabela do Wesley e com D1 a D4 (lista de checagem em `conferencia_registro.md`).
- [ ] Copiar para `outputs/sprint5/` os arquivos da rodada com as previsões oficiais (`metrics_oficial*.csv` e `run_meta_oficial*.json`), se ela foi feita em outra máquina.
- [ ] v3 do modelo: deslocamento maior na augmentation (a AUC cai 0,07 a 16 mm) e mais sementes.
- [ ] Explicabilidade: reportar o Pointing Game **no recorte deslocado** ao lado do mapa centrado, em vez do IoU absoluto.
- [ ] Pleura: validar a segmentação do pulmão numa amostra e repetir a oclusão com preenchimento mais realista (interpolação em vez de valor constante).
- [ ] Usar a união das máscaras dos leitores e comparar com a análise da Débora.
