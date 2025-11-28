- Pergunta: Qual é o problema e quais os tipos de dados estão disponíveis?
  - Resposta: O problema é classificar imagens de lesões de pele em duas categorias: melanoma e não-melanoma. Os dados disponíveis são imagens dermatoscópicas rotuladas com 0 (lesão benigna) ou 1 (lesão maligna).

- Pergunta: Quais as características disponíveis e como elas estão distribuídas?
  - Resposta: As características disponíveis são as próprias imagens, que carregam informações visuais como textura, cor, padrões irregulares, bordas e assim por diante. Quando usamos apenas imagem, essas características não são explícitas: o modelo aprende representações automaticamente. A distribuição das classes é bem desbalanceada: existem muito mais imagens de lesões benignas do que de melanoma.

- Pergunta: Que tipo de problema iremos lidar? Regressão, classificação, etc.?
  - Resposta: O tipo de problema é claramente de classificação binária: dado uma imagem, queremos prever se ela corresponde a melanoma ou não.

- Pergunta: Qual a relação das características entre si e com a variável alvo? Elas estão correlacionadas?
  - Resposta: As relações entre características visuais entre si e com a variável-alvo não são triviais de medir diretamente, porque são pixels e não features numéricas. Porém, sabe-se que mudanças de textura, assimetria, irregularidade e variações de cor são indicativos de melanoma. Em modelos que extraem embeddings (CNNs ou Transformers), as representações aprendidas costumam mostrar alguma separação entre classes, mas isso é descoberto durante o treinamento.

- Pergunta: É um problema desbalanceado?
  - Resposta: O problema é fortemente desbalanceado, então, sem técnicas adequadas (pesos de classe, oversampling, métricas apropriadas), o modelo tende a prever a classe majoritária.

- Pergunta: Os dados precisarão de algum tipo de pré-processamento?
  - Resposta: O pré-processamento necessário inclui: normalizar os pixels, possivelmente aplicar data augmentation (para aumentar diversidade e lidar com o desbalanceamento). O dataset já tem as imagens redimensionadas e é dividido entre treino/validação/teste, de maneira que são etapas não necessárias.

- Pergunta: Quais as métricas utilizadas neste problema? Acurácia, erro, MSRE, etc.? O que elas significam?
  - Resposta: As métricas que vamos usar são:
    - Acurácia balanceada: é a média do recall de cada classe. É útil quando existe desbalanceamento porque dá o mesmo peso para melanoma e não-melanoma.
    - F-beta com beta=2: essa métrica dá mais peso ao recall do que à precisão. Nós escolhemos beta=2 porque é pior deixar de detectar um melanoma do que gerar um falso positivo. Assim, o modelo é incentivado a errar menos quando a classe é melanoma.