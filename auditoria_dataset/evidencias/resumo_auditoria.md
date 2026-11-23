# Resumo da auditoria do dataset oficial (Sprint 5, P3)

Rodado em 2026-10-26 20:31.

| Seção | Status | O que significa |
|---|---|---|
| 1. Separação | PENDENTE | Sem o arquivo da divisão, não dá para conferir a separação. |
| 2. Composição | PENDENTE | Sem o arquivo de previsões (com paciente), não dá para montar a composição. |
| 3. Descartes | PENDENTE | Funil sem a divisão oficial: só mostra o que existe no LIDC. |
| 3. Descartes | PENDENTE | Sem o arquivo de previsões, a correspondência recorte -> paciente -> rótulo fica pendente. |
| 4. Origem dos 48 | INFO | O piloto tem 48 pacientes (0001 a 0050, sem [28, 32]); o cruzamento com os folds oficiais fica PENDENTE. |
| 5. Explicabilidade | PENDENTE | Lista 'xai_160' não encontrada (casos_xai.csv). |
| 5. Explicabilidade | PENDENTE | Lista 'adebayo_24' não encontrada (casos_adebayo.csv). |
| 5. Explicabilidade | PENDENTE | Lista 'exemplos_16' não encontrada (exemplos.csv). |
| 6. D6 e tamanho | PENDENTE | Sem a média de malignidade nas previsões, não dá para mostrar o efeito da regra. |
| 6. D6 e tamanho | INFO | Tamanho calculado das anotações originais (7034 anotações, em mm), uma linha por radiologista. É um diâmetro por fatia, não o diâmetro 3D oficial. Melhor: o André exportar a coluna de diâmetro do pipeline dele. |
| 7. Vazamento (D5) | ALERTA | Só com a ficha do radiologista, sem ver a imagem, um modelo simples chega a AUC 0.908 (531 pacientes). A ficha praticamente entrega a resposta: MLP semântico e fusão devem ficar como 'não comparáveis' com o modelo oficial (só imagem). |
| 7. Vazamento (D5) | INFO | Teste feito sem a divisão oficial (grupos fixos pelo número do paciente): resultado provisório. |
