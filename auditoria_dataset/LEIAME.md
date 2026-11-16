# Auditoria do dataset — execução na máquina do Gabriel (26/10)

O notebook `Auditoria_Dataset_Oficial.ipynb` é a auditoria do Matheus, executado do início ao fim no dataset completo desta máquina (`D:\datasetLIDC_completo\LIDC-IDRI`). As saídas das células ficaram salvas no próprio notebook, e as evidências estão em `evidencias/`. O log da execução está em `execucao_log.txt`.

## Adaptações feitas na cópia (marcadas com `[Adaptação Gabriel]` no código)

Nesta máquina, os 1.319 XMLs ficam todos em `_xml_official/` (o pacote oficial do TCIA). O caminho deles não traz o ID do paciente, e não há DICOM na mesma pasta. O código original tirava o paciente do caminho do XML, então sem ajuste nenhum XML teria paciente. A cópia foi adaptada assim:

1. **Configuração:** `PASTA_LIDC` aponta para o dataset completo, e `INDICE_DICOM` aponta para `outputs/sprint3/dicom_index.csv`.
2. **Seção 3:** o paciente é identificado pela série do XML no índice DICOM e, se a série não estiver lá, pelo exame.
3. **Seção 6:** o tamanho do pixel é lido de um DICOM da série, e não de um DICOM na pasta do XML.

O resto do código está igual ao original. A pasta `dados_andre/` (divisão, previsões oficiais e listas de casos de explicabilidade do André) **não existe nesta máquina**, então as seções 1, 2, 4 e 5 ficaram PENDENTE.

## Resultados

**O que existe no disco** (conferido fora do notebook):

| Item | Pacientes |
|---|---|
| Pastas de paciente | 1.010 |
| Pastas com algum DICOM | 709 |
| Pacientes com TC (no índice DICOM) | 642 |
| Pastas só com radiografia (DX/CR, fora do índice de TC, como esperado) | 67 |
| Pastas vazias | 301 |

**Funil da auditoria** (`evidencias/3_funil_pacientes.csv`):

| Etapa | Pacientes |
|---|---|
| Pastas de paciente no LIDC | 1.010 |
| Pacientes com alguma anotação (XML ligado a uma TC desta máquina) | 627 |
| Pacientes com nódulo ≥ 3 mm com nota de malignidade | **559** |

Os **559 pacientes** com nódulo com nota são exatamente os 559 da divisão auditada (391/84/84). A divisão aproveitou todos os pacientes elegíveis que existem nesta máquina. Dos XMLs, 2.482 anotações não puderam ser ligadas a um paciente, porque pertencem a pacientes sem TC no disco.

**Tamanho dos nódulos** (`evidencias/6_tamanho_nodulos_por_anotacao.csv`, para a Débora): 4.552 anotações com nota, de 559 pacientes. O diâmetro equivalente da maior ROI tem mediana de 6,9 mm (de 2,7 a 67,5 mm), e 75% das anotações ficam abaixo de 10,3 mm. É um diâmetro por fatia e por leitor, não o diâmetro 3D.

**D5: a ficha do radiologista entrega a resposta?** (`evidencias/7_vazamento_ficha_radiologista.csv`). Sem nenhuma imagem, só com as 8 características da ficha, uma regressão logística prevê a nota do mesmo radiologista com:
- **AUC 0,908** na visão principal (sem nota 3; 2.773 anotações, 531 pacientes);
- **AUC 0,827** com todos (nota 3 = benigno; 4.552 anotações, 559 pacientes).

As características que mais carregam a resposta são lobulação (0,742), espiculação (0,740) e sutileza (0,731). **ALERTA:** a ficha praticamente entrega o rótulo. Por isso, o MLP semântico e a fusão têm que aparecer como **não comparáveis** com os modelos que só veem a imagem. Esse resultado é provisório, porque os grupos foram fixados pelo número do paciente, sem a divisão oficial.

## Pendências (dependem de arquivos do André)

Para completar as seções 1, 2, 4 e 5, colocar em `auditoria_dataset/dados_andre/`:
- `divisao_pacientes.csv`;
- `previsoes_oficiais.csv`;
- `casos_xai.csv`, `casos_adebayo.csv` e `exemplos.csv`.

Depois, rodar de novo:

```text
.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace auditoria_dataset\Auditoria_Dataset_Oficial.ipynb
```

Atenção: a divisão do André tem 757 pacientes e vem de outra base (SEG/SR no Kaggle). Nesta máquina existem só 642 pacientes com TC, então parte dos pacientes dele não vai estar aqui. O próprio notebook avisa isso como INFO.
