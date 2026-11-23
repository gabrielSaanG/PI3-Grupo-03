# Sprint Review 5 — Gabriel (resumo para os slides)

Fonte de todos os números: `docs/sprint5/gabriel_validacao_estatistica.md`. **Todos são provisórios.** Vêm de modelos de semente única treinados por mim na divisão auditada, não do ensemble oficial do André.

---

## Slide 1 — Objetivo da minha parte

- Medir o modelo com métricas adequadas a um teste desbalanceado.
- Calcular intervalos de confiança de 95% **por paciente**, não por patch.
- Comparar com referências que não aprendem nada (o "sempre maligno") e com uma CNN simples, **nos mesmos casos**.
- Verificar o teste com rótulos embaralhados.
- Garantir uma única divisão: **559 pacientes (391/84/84), seed 42**.

---

## Slide 2 — O que foi verificado antes de calcular

- A divisão foi reproduzida e bateu exatamente com a auditoria: 18.508 / 3.694 / 4.163 patches.
- As previsões do ResNet50 foram geradas de novo, caso a caso, com ID do paciente e do nódulo.
- A matriz de confusão saiu **idêntica** à registrada: TN 528, FP 496, FN 756, TP 2.383. Cada previsão está ligada ao rótulo certo.
- Unidade de análise: **nódulo** (média dos patches). O bootstrap reamostra **pacientes** (1.000 vezes).

---

## Slide 3 — A acurácia engana (modelo do notebook, rótulo antigo)

| Por nódulo (687, 84 pacientes) | Sempre maligno | CNN simples | ResNet50 v1 |
|---|---|---|---|
| AUC | 0,500 | 0,611 | **0,712 [0,651; 0,776]** |
| Acurácia balanceada | 0,500 | 0,528 | **0,627 [0,575; 0,685]** |
| Acurácia | **0,702** | 0,370 | 0,674 |

- O "sempre maligno" acerta mais (70%) que o ResNet50 (67%) **sem aprender nada**.
- Os "70%" dos slides antigos eram a acurácia por patch (0,699). Esse número não deve ser o principal.
- O limiar escolhido na validação **não ajudou** o v1: a acurácia balanceada continuou 0,627. O problema era separar as classes, não o limiar.

---

## Slide 4 — Retreino: ResNet50 v2 (mesma divisão)

Mudanças, decididas antes de olhar o teste:
1. **Rótulo de consenso:** média dos leitores do mesmo nódulo ≥ 3,5 (antes, cada leitor ≥ 3, e 3 = "indeterminado").
2. **Ambíguos** (média entre 2,5 e 3,5) fora do treino. No teste, os resultados aparecem com e sem eles.
3. **Recorte fixo de 64 mm** (antes, cada caixa era esticada para 224 px, e o tamanho se perdia).
4. **Augmentation** (rotação, flip, deslocamento) e **época escolhida pela AUC da validação**.

Figura sugerida: `docs/sprint5/figures/recortes_v1_v2.png`, com os recortes antes e depois.

---

## Slide 5 — Resultado do v2 (provisório)

Teste com **todos** os nódulos (84 pacientes, 26% malignos no rótulo novo):

| | Sempre maligno | v1 (mesmo rótulo) | **v2** |
|---|---|---|---|
| AUC | 0,500 | 0,818 | **0,906 [0,852; 0,948]** |
| Acurácia balanceada | 0,500 | 0,710 | **0,812 [0,747; 0,866]** |
| Malignos perdidos | 0 | 4 | 25 |
| Benignos marcados por engano | 508 | 283 | 120 |

- Sem os ambíguos (72 pacientes): AUC **0,949** e acurácia balanceada 0,880. É uma tarefa mais fácil, e por isso sempre mostramos as duas visões.
- **Atenção:** só a troca de rótulo já leva o v1 de 0,712 para 0,818. A tarefa mudou, e isso depende da decisão do grupo (D6).

---

## Slide 6 — Explicabilidade no v2 (versão mínima)

Teste: 275 nódulos, quatro mapas e limiar escolhido na validação.

| Mapa | IoU | Pointing Game |
|---|---|---|
| Grad-CAM | 0,11 | 42% |
| Integrated Gradients | 0,16 | 66% |
| Aleatório | 0,02 | 2% |
| **Centrado** (não aprende nada) | **0,50** | **100%** |

- O mapa centrado **vence**, porque todo recorte tem o nódulo no centro. Essas métricas **não provam** que a rede olha para o nódulo.
- O Grad-CAM marca com frequência **a pleura ou a parede torácica**. Pode ser um atalho de contexto, e isso será investigado.
- Por tamanho: em nódulos com 10 mm ou mais, o Grad-CAM acerta de 72% a 95%; abaixo de 10 mm, cerca de 30%.
- A meta de IoU ≥ 0,40, sozinha, não serve: o mapa centrado passa dela.

Figura sugerida: `docs/sprint5/figures/xai_v2_exemplos.png`.

---

## Slide 7 — Teste com rótulos embaralhados (o que havia nos arquivos)

- **O que foi embaralhado:** os rótulos de treino e de validação, por patch. Um ResNet50 novo foi treinado com eles.
- **O que se esperava:** desempenho no nível do acaso e mapas diferentes.
- **O que aconteceu:**
  - os mapas mudaram, como esperado (Spearman mediana de 0,03);
  - **mas** o modelo com rótulos aleatórios acertou 62% contra 51% do original, nos mesmos 100 casos.
- **Leitura:** isso é **um alerta, não uma aprovação**. Além disso, o resultado é da **rodada pequena antiga**, não da divisão de 559. Precisa ser refeito.

---

## Slide 8 — Limitações

- Tudo é provisório: uma semente, sem ensemble e sem TTA.
- O teste tem só 84 pacientes, então os intervalos são largos.
- O rótulo é a opinião dos radiologistas, não biópsia.
- A unidade é o nódulo de cada leitor (o mesmo nódulo físico pode contar mais de uma vez).
- Os baselines antigos (teste de 90 casos) **não são comparáveis**.
- O notebook do André usa **outra base** (757 pacientes, validação cruzada, rótulo ≥ 3,5) e **não é a divisão auditada**. Os números dele não podem ser misturados com estes.

---

## Slide 9 — Pendências para a Sprint 6

1. **Decisão do grupo:** qual rótulo (consenso ou por leitor) e qual base (559 ou 757) são os oficiais. Registrar na tabela do Wesley.
2. Previsões oficiais do André na divisão auditada, para rodar o mesmo script com `--status oficial`.
3. v2: mais seeds ou ensemble (a melhor época foi a 1, sinal de que decora rápido) e recortes de vários tamanhos.
4. Explicabilidade:
   - teste de Adebayo no v2;
   - teste com o nódulo deslocado do centro;
   - investigar a pleura;
   - usar a união das máscaras dos leitores;
   - alinhar com a análise da Débora.
5. Calibração (ECE) do v2 (João).
6. Conferir slides e documentos contra a tabela de registro (um número oficial só).

---

## Frase de fechamento

> Tiramos a acurácia do centro da avaliação, medimos com intervalos por paciente e comparamos com referências que não aprendem nada. Na mesma divisão auditada, o modelo retreinado separa bem as classes (AUC 0,91). Mas a explicabilidade ainda não prova que a rede olha para o nódulo: o mapa centrado ganha. Isso está registrado como resultado, junto com as limitações.
