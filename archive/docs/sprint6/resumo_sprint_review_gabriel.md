# Sprint Review 6 — Gabriel (resumo para os slides)

Fonte de todos os números: `docs/sprint6/gabriel_sprint6.md`. **Todos são provisórios:** ResNet50 v2 de uma semente, divisão auditada de 559 pacientes (391/84/84), rótulo de consenso (média ≥ 3,5).

---

## Slide 1 — O que fiz nesta sprint

- Reunião de decisões D1 a D4 (04/11), com as respostas registradas na página.
- Métricas com IC por paciente ao lado do "sempre maligno".
- **Conferência dos números** do repositório antes da Review 6.
- **Teste de Adebayo** e **teste de deslocamento** no v2.
- **Pleura e parede torácica:** o modelo usa o contexto como atalho?

---

## Slide 2 — Adebayo: randomização dos pesos

Os blocos do v2 são reinicializados de cima para baixo, e os mapas são comparados com os do v2 (275 nódulos do teste).

| | v2 | Só o `fc` aleatório | Rede inteira aleatória |
|---|---|---|---|
| Grad-CAM: Spearman com o v2 | 1 | −0,02 | −0,04 |
| Grad-CAM: Pointing Game | 42% | 10% | 7% |
| IG: Spearman com o v2 | 1 | 0,77 | 0,43 |
| IG: Pointing Game | 66% | 45% | 4% |

- **Grad-CAM passa:** o mapa muda logo na primeira etapa.
- **IG passa no pico, não no mapa:** o pico sai do nódulo quando a rede é aleatorizada, mas o mapa continua parecido, porque o IG acompanha o brilho da imagem.

Figura sugerida: `docs/sprint6/figures/adebayo_v2_curvas.png`.

---

## Slide 3 — Adebayo: rótulos embaralhados (refeito na divisão de 559)

Rótulos permutados **por nódulo físico** e modelo treinado com a mesma receita do v2. O modelo decorou 95% dos rótulos aleatórios do treino.

| Teste completo (687 nódulos, 84 pacientes) | Sempre maligno | v2 | Rótulos embaralhados |
|---|---|---|---|
| AUC | 0,500 | **0,906** [0,852; 0,948] | **0,555** [0,464; 0,652] |
| Acurácia balanceada | 0,500 | 0,812 | 0,562 |
| Acurácia | 0,261 | 0,789 | 0,623 |

- O modelo com rótulos aleatórios fica **no acaso**. O alerta da Sprint 3 (62% contra 51%) vinha da amostra pequena e da permutação por patch.
- A acurácia engana de novo: 62% sem aprender nada, só respondendo "benigno".
- Pointing Game do Grad-CAM: 42% no v2 contra 9% no modelo aleatório.

---

## Slide 4 — Teste de deslocamento: o nódulo fora do centro

Antes, o mapa centrado (que não aprende nada) vencia, porque o nódulo estava sempre no centro. Agora, o recorte é deslocado.

| Pointing Game | 0 mm | 8 mm | 16 mm |
|---|---|---|---|
| Grad-CAM | 42% | 38% | **27%** |
| Integrated Gradients | 66% | 60% | **37%** |
| Centrado | 100% | 43% | **3%** |
| AUC do v2 | 0,93 | 0,92 | 0,86 |

- **Com o nódulo fora do centro, o Grad-CAM e o IG vencem o centrado.** Os mapas localizam de verdade, não só pelo viés do centro.
- Mas a localização piora: nos nódulos menores que 10 mm, o acerto fica abaixo de 35% a 16 mm.
- O modelo também perde AUC fora do centro (0,93 para 0,86), um argumento para mais deslocamento no treino.

Figura sugerida: `docs/sprint6/figures/deslocamento_v2_curvas.png`.

---

## Slide 5 — Pleura e parede torácica: é atalho?

- O pico do Grad-CAM cai fora do pulmão em **28%** dos casos, sobretudo em nódulos colados à pleura (**45%**) e quando o modelo prevê **benigno** (**37%**, contra 11% no maligno).
- **Teste de oclusão** (147 nódulos não ambíguos):

| Condição | AUC |
|---|---|
| Original | 0,932 |
| Sem o que está fora do pulmão | **0,933** |
| Só o nódulo | **0,929** |
| Sem o nódulo | **0,756** |

- **Não é atalho:** apagar a parede torácica não muda a AUC, e a proximidade da pleura não prevê o rótulo (AUC 0,48). A separação vem do nódulo.
- **Mas o contexto empurra para "benigno":** sem ele, as probabilidades sobem e 11% das previsões trocam de classe. Isso afeta o limiar e a calibração, e fica registrado como risco.

Figura sugerida: `docs/sprint6/figures/pleura_v2_exemplos.png`.

---

## Slide 6 — Conferência dos números antes da Review 6

No repositório, há:
- **três bases:** 559 pacientes auditados, 757 do André e rodadas antigas (27 nódulos; 90 casos). Há também o IQ-OTH/NCCD, que é outro dataset (97%);
- **três regras de rótulo:** ≥ 3 por leitor (v1), média ≥ 3,5 (v2) e ≥ 4 (teste da ficha, D5);
- **oito acurácias diferentes** na mesma divisão, de 0,58 a 0,88.

Regra para a Review 6: **uma base, um rótulo e uma acurácia**, iguais à tabela de registro, sempre com AUC, acurácia balanceada e o "sempre maligno" ao lado.

---

## Slide 7 — Limitações

- Tudo é provisório: uma semente, sem ensemble e sem TTA.
- Os testes de explicabilidade usam um patch por nódulo e a máscara de um leitor.
- A oclusão cria imagens artificiais, e a segmentação do pulmão é heurística (falha em alguns cortes).
- A AUC por nódulo físico desta análise (0,93) **não é** o número oficial do v2 (0,949 sem ambíguos, ou 0,906 com todos).

---

## Slide 8 — Próximos passos

1. Comparar cada slide com a tabela do Wesley e com as decisões D1 a D4.
2. v3 do modelo: mais deslocamento na augmentation e mais sementes.
3. Reportar a explicabilidade **no recorte deslocado**, ao lado do mapa centrado.
4. Validar a segmentação do pulmão e repetir a oclusão com preenchimento realista.
5. União das máscaras dos leitores e alinhamento com a análise da Débora.

---

## Frase de fechamento

> O v2 passou nos dois testes de sanidade: com rótulos embaralhados ele fica no acaso, e o Grad-CAM muda quando os pesos são aleatorizados. Com o nódulo fora do centro, o Grad-CAM e o IG passam a vencer o mapa centrado. A parede torácica que o Grad-CAM marca não é o que separa as classes, mas empurra as probabilidades para "benigno". E, antes da Review 6, mapeamos três bases, três rótulos e oito acurácias que precisam virar um número só.
