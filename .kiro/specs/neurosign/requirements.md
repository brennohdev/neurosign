# Documento de Requisitos — NeuroSign

## Introdução

O NeuroSign é uma aplicação web de tradução em tempo real de Língua de Sinais Americana (ASL) para texto e áudio. O sistema captura landmarks das mãos no cliente via MediaPipe, transmite tensores numéricos leves por WebSocket para um backend FastAPI, que executa inferência com um modelo LSTM bidirecional com atenção exportado em ONNX quantizado (Int8). O objetivo principal é demonstrar domínio em ML Engineering end-to-end — desde coleta e treinamento até deploy — como projeto de portfólio.

O monorepo é organizado em dois espaços de trabalho principais: `/ml-lab` (pesquisa e treinamento) e `/apps` (produção), gerenciados com `uv`.

---

## Glossário

- **Sistema**: O conjunto completo da aplicação NeuroSign (frontend + backend + modelo).
- **Cliente**: A aplicação React/TypeScript executada no navegador do usuário.
- **Backend**: O serviço FastAPI responsável por inferência e orquestração.
- **Landmark_Extractor**: O componente MediaPipe (WebAssembly) no Cliente que detecta e extrai landmarks das mãos.
- **Landmark_Normalizer**: O componente do Cliente que normaliza landmarks em relação ao pulso (wrist) e escala pelo tamanho da mão.
- **WebSocket_Client**: O componente do Cliente responsável pela conexão e envio de tensores ao Backend.
- **WebSocket_Server**: O componente do Backend que recebe tensores e gerencia sessões.
- **Sliding_Window**: O componente do Backend que acumula tensores em janelas temporais com hiperparâmetros configuráveis (número de frames e stride).
- **Inference_Engine**: O componente do Backend que executa inferência usando ONNX Runtime com modelo quantizado Int8.
- **Prediction_Broadcaster**: O componente do Backend que transmite predições de volta ao Cliente via WebSocket.
- **ML_Pipeline**: O conjunto de scripts e notebooks em `/ml-lab` responsáveis por treinamento, validação e exportação do modelo.
- **WLASL_Dataset**: O dataset Word-Level American Sign Language do Kaggle, utilizado para treinamento.
- **ONNX_Model**: O modelo LSTM bidirecional com atenção exportado em formato ONNX com quantização Float32 → Int8.
- **Top-1 Accuracy**: Percentual de predições em que o sinal correto é o mais provável.
- **Top-5 Accuracy**: Percentual de predições em que o sinal correto está entre os 5 mais prováveis.
- **Latência_ONNX**: Tempo de execução da inferência no ONNX Runtime, medido em milissegundos.
- **P50/P95**: Percentis 50 e 95 de latência, usados como métricas de benchmark.

---

## Requisitos

### Requisito 1: Captura e Normalização de Landmarks no Cliente

**User Story:** Como usuário, quero que minha câmera seja processada localmente no navegador, para que meus dados biométricos não sejam transmitidos pela rede.

#### Critérios de Aceitação

1. WHEN o usuário concede permissão de câmera, THE Landmark_Extractor SHALL detectar landmarks de até duas mãos em cada frame de vídeo usando MediaPipe Hands via WebAssembly.
2. WHEN landmarks são detectados, THE Landmark_Normalizer SHALL transladar todos os pontos usando o ponto do pulso (landmark 0) como origem, resultando em coordenadas relativas.
3. WHEN landmarks são normalizados pela origem, THE Landmark_Normalizer SHALL escalar as coordenadas pelo tamanho da mão, definido como a distância euclidiana entre o pulso (landmark 0) e a base do dedo médio (landmark 9).
4. IF nenhuma mão for detectada em um frame, THEN THE Landmark_Extractor SHALL emitir um tensor de zeros com as mesmas dimensões esperadas pelo Sliding_Window.
5. THE Landmark_Extractor SHALL processar frames a uma taxa mínima de 15 FPS em hardware de consumo moderno (CPU de 4 núcleos, sem GPU dedicada).

---

### Requisito 2: Transmissão de Tensores via WebSocket

**User Story:** Como desenvolvedor, quero transmitir apenas tensores numéricos leves pela rede, para minimizar o uso de banda e latência de transmissão.

#### Critérios de Aceitação

1. WHEN landmarks normalizados estão disponíveis, THE WebSocket_Client SHALL serializar o tensor como array JSON e enviá-lo ao WebSocket_Server.
2. THE WebSocket_Client SHALL manter uma única conexão WebSocket persistente durante toda a sessão de uso.
3. IF a conexão WebSocket for interrompida, THEN THE WebSocket_Client SHALL tentar reconectar automaticamente com backoff exponencial, com intervalo inicial de 1 segundo e máximo de 30 segundos.
4. IF o WebSocket_Server rejeitar a conexão, THEN THE WebSocket_Client SHALL exibir uma mensagem de erro descritiva ao usuário.
5. THE WebSocket_Server SHALL aceitar conexões simultâneas de múltiplos clientes de forma independente, sem compartilhar estado de sessão entre eles.

---

### Requisito 3: Acumulação em Sliding Window

**User Story:** Como engenheiro de ML, quero acumular frames em janelas temporais configuráveis, para que o modelo receba sequências com contexto temporal suficiente para reconhecer sinais.

#### Critérios de Aceitação

1. WHEN um tensor é recebido pelo WebSocket_Server, THE Sliding_Window SHALL adicioná-lo ao buffer da sessão correspondente.
2. THE Sliding_Window SHALL ser configurado com dois hiperparâmetros explícitos: `window_size` (número de frames por janela) e `stride` (número de frames entre janelas consecutivas), definidos via variáveis de ambiente.
3. WHEN o buffer acumular `window_size` frames, THE Sliding_Window SHALL emitir a janela completa ao Inference_Engine.
4. WHEN uma janela é emitida, THE Sliding_Window SHALL avançar o buffer em `stride` frames, descartando os frames mais antigos.
5. WHILE o buffer contiver menos frames que `window_size`, THE Sliding_Window SHALL reter os tensores sem emitir janelas ao Inference_Engine.

---

### Requisito 4: Inferência com ONNX Runtime

**User Story:** Como usuário, quero receber a tradução do sinal em tempo real, para que a comunicação seja fluida e sem atrasos perceptíveis.

#### Critérios de Aceitação

1. WHEN uma janela completa é recebida, THE Inference_Engine SHALL executar inferência usando o ONNX_Model carregado em memória.
2. THE Inference_Engine SHALL retornar as 5 predições mais prováveis (Top-5) com seus respectivos scores de confiança.
3. THE Inference_Engine SHALL executar cada inferência com latência P50 inferior a 50ms e P95 inferior a 100ms, medidos em CPU sem aceleração de hardware dedicada.
4. IF o ONNX_Model não for encontrado no caminho configurado durante a inicialização, THEN THE Inference_Engine SHALL encerrar o processo com uma mensagem de erro descritiva.
5. THE Inference_Engine SHALL carregar o ONNX_Model uma única vez na inicialização do Backend e reutilizá-lo para todas as inferências subsequentes.

---

### Requisito 5: Transmissão de Predições ao Cliente

**User Story:** Como usuário, quero ver o texto traduzido aparecer na tela em tempo real, para acompanhar a tradução enquanto realizo os sinais.

#### Critérios de Aceitação

1. WHEN o Inference_Engine produz uma predição, THE Prediction_Broadcaster SHALL transmitir o resultado ao Cliente correspondente via WebSocket em formato JSON contendo o sinal predito e o score de confiança.
2. THE Cliente SHALL exibir o sinal com maior score de confiança como tradução principal na interface.
3. WHERE o usuário habilitar o modo de exibição expandida, THE Cliente SHALL exibir as 5 predições Top-5 com seus respectivos scores de confiança.
4. WHEN uma nova predição é recebida, THE Cliente SHALL atualizar a exibição em até 100ms após o recebimento da mensagem WebSocket.
5. WHERE o usuário habilitar síntese de voz, THE Cliente SHALL converter o sinal predito em áudio usando a Web Speech API do navegador.

---

### Requisito 6: Pipeline de ML — Preparação de Dados

**User Story:** Como engenheiro de ML, quero um pipeline reprodutível de preparação de dados, para garantir que o treinamento seja consistente e rastreável.

#### Critérios de Aceitação

1. THE ML_Pipeline SHALL baixar e extrair o WLASL_Dataset do Kaggle usando credenciais configuradas via variáveis de ambiente.
2. THE ML_Pipeline SHALL filtrar o WLASL_Dataset para os top-50 sinais por frequência de amostras, descartando os demais.
3. THE ML_Pipeline SHALL aplicar a mesma normalização de landmarks (origem no pulso, escala pelo tamanho da mão) definida no Requisito 1 a todos os exemplos do dataset.
4. THE ML_Pipeline SHALL dividir o dataset em conjuntos de treino, validação e teste com proporções configuráveis, garantindo que exemplos do mesmo sinal não apareçam em múltiplos conjuntos.
5. THE ML_Pipeline SHALL serializar os conjuntos processados em formato compatível com PyTorch DataLoader para uso no treinamento.

---

### Requisito 7: Pipeline de ML — Treinamento do Modelo

**User Story:** Como engenheiro de ML, quero treinar um modelo LSTM bidirecional com atenção localmente, para validar a arquitetura antes de considerar modelos mais complexos.

#### Critérios de Aceitação

1. THE ML_Pipeline SHALL treinar um modelo LSTM bidirecional com mecanismo de atenção usando PyTorch, com suporte a aceleração via MPS (Apple Silicon).
2. WHEN o treinamento for concluído, THE ML_Pipeline SHALL reportar as métricas Top-1 Accuracy e Top-5 Accuracy no conjunto de validação.
3. THE ML_Pipeline SHALL salvar checkpoints do modelo ao final de cada época, retendo apenas os 3 melhores checkpoints por Top-1 Accuracy no conjunto de validação.
4. THE ML_Pipeline SHALL registrar métricas de treinamento (loss, Top-1, Top-5) por época em formato compatível com TensorBoard ou similar.
5. IF o treinamento for interrompido, THEN THE ML_Pipeline SHALL permitir retomada a partir do último checkpoint salvo.

---

### Requisito 8: Pipeline de ML — Exportação e Quantização ONNX

**User Story:** Como engenheiro de ML, quero exportar o modelo treinado para ONNX com quantização Int8, para garantir inferência eficiente no backend de produção.

#### Critérios de Aceitação

1. WHEN o melhor checkpoint é selecionado, THE ML_Pipeline SHALL exportar o modelo para formato ONNX com opset compatível com ONNX Runtime.
2. THE ML_Pipeline SHALL aplicar quantização estática Float32 → Int8 ao modelo ONNX exportado.
3. THE ONNX_Model quantizado SHALL ter tamanho de arquivo inferior a 10MB.
4. WHEN a quantização é concluída, THE ML_Pipeline SHALL executar um benchmark de inferência reportando latência P50 e P95 em CPU, usando um conjunto de 100 amostras do conjunto de teste.
5. THE ML_Pipeline SHALL validar que a diferença de Top-1 Accuracy entre o modelo Float32 e o modelo Int8 quantizado é inferior a 2 pontos percentuais no conjunto de teste.

---

### Requisito 9: Infraestrutura e Reprodutibilidade

**User Story:** Como desenvolvedor, quero executar toda a aplicação com um único comando, para facilitar o desenvolvimento local e a demonstração do portfólio.

#### Critérios de Aceitação

1. THE Sistema SHALL fornecer um arquivo `docker-compose.yml` que inicialize o Backend e seus serviços dependentes com o comando `docker-compose up`.
2. THE Sistema SHALL gerenciar todas as dependências Python do monorepo usando `uv` com workspaces, separando as dependências de `/ml-lab` e `/apps`.
3. THE Backend SHALL expor variáveis de ambiente documentadas para todos os hiperparâmetros configuráveis, incluindo `window_size`, `stride` e o caminho do ONNX_Model.
4. THE Sistema SHALL incluir um arquivo `README.md` com diagrama de fluxo de dados, instruções de setup, e um GIF ou vídeo demonstrativo da aplicação em funcionamento.
5. IF uma variável de ambiente obrigatória não estiver definida na inicialização do Backend, THEN THE Backend SHALL encerrar o processo com uma mensagem de erro listando todas as variáveis ausentes.

---

### Requisito 10: Qualidade de Código e Arquitetura

**User Story:** Como desenvolvedor, quero que o backend siga Clean Architecture com Ports & Adapters, para que a lógica de negócio seja independente de frameworks e facilmente testável.

#### Critérios de Aceitação

1. THE Backend SHALL separar a lógica de inferência em uma camada de domínio independente de FastAPI, ONNX Runtime e WebSocket, acessível apenas via interfaces (ports) definidas explicitamente.
2. THE Backend SHALL implementar adaptadores concretos para ONNX Runtime e WebSocket que satisfaçam as interfaces de domínio sem expor detalhes de implementação às camadas internas.
3. THE ML_Pipeline SHALL separar código de experimentos (notebooks) de código de produção (módulos Python importáveis) dentro do diretório `/ml-lab`.
4. THE Sistema SHALL incluir testes unitários para a lógica de normalização de landmarks, cobrindo os casos: detecção normal, mão ausente (tensor de zeros) e landmarks em posições extremas.
5. THE Sistema SHALL incluir testes de integração para o fluxo WebSocket completo: envio de tensor → sliding window → inferência → resposta, usando o ONNX_Model real ou um modelo stub de dimensões equivalentes.
