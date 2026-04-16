# Plano de Implementação: NeuroSign

## Visão Geral

Implementação incremental do NeuroSign em três frentes paralelas: ML Lab (pipeline de dados, treinamento e exportação), Backend (Clean Architecture com FastAPI + ONNX Runtime) e Frontend (React + MediaPipe + WebSocket). As tarefas são ordenadas para que cada etapa produza código integrável à anterior.

## Tarefas

- [x] 1. Estrutura do monorepo e configuração base
  - Criar `pyproject.toml` raiz com workspaces `uv` apontando para `ml-lab` e `apps/backend`
  - Criar `apps/backend/pyproject.toml` com dependências: `fastapi`, `uvicorn`, `onnxruntime`, `numpy`, `python-dotenv`, `hypothesis`, `pytest`, `pytest-asyncio`, `httpx`
  - Criar `ml-lab/pyproject.toml` com dependências: `torch`, `onnx`, `onnxruntime`, `kaggle`, `hypothesis`, `pytest`
  - Criar `apps/frontend/package.json` com dependências: `react`, `typescript`, `vite`, `@mediapipe/hands`, `fast-check`, `vitest`, `@testing-library/react`
  - Criar estrutura de diretórios conforme design: `apps/backend/neurosign_backend/{domain,application,adapters}`, `ml-lab/neurosign_ml/{data,models,training,export}`, `apps/frontend/src/{hooks,components,lib}`
  - _Requisitos: 9.2_

- [x] 2. Configuração e entidades de domínio do backend
  - [x] 2.1 Implementar `EnvConfig` em `apps/backend/neurosign_backend/adapters/config.py`
    - Ler `WINDOW_SIZE`, `STRIDE`, `MODEL_PATH`, `HOST`, `PORT` via `os.environ`
    - Coletar todas as variáveis ausentes e encerrar com mensagem listando cada uma
    - _Requisitos: 9.3, 9.5_
  - [ ]* 2.2 Escrever teste de propriedade para `EnvConfig` (Propriedade 9)
    - **Propriedade 9: Validação de configuração lista todas as variáveis ausentes**
    - **Valida: Requisito 9.5**
  - [ ]* 2.3 Escrever testes unitários para `EnvConfig`
    - Caso: todas as variáveis presentes; caso: variáveis ausentes individualmente e em conjunto
    - _Requisitos: 9.5_
  - [x] 2.4 Definir entidades de domínio em `apps/backend/neurosign_backend/domain/entities.py`
    - `Prediction(label, confidence, rank)` e `InferenceResult(predictions, latency_ms)` como dataclasses frozen
    - _Requisitos: 4.2, 5.1_
  - [x] 2.5 Definir ports em `apps/backend/neurosign_backend/domain/ports.py`
    - `InferencePort(Protocol)` com `predict(window) -> list[Prediction]`
    - `SessionPort(Protocol)` com `add_frame(session_id, frame) -> Optional[np.ndarray]`
    - _Requisitos: 10.1, 10.2_

- [x] 3. Sliding Window (camada de aplicação do backend)
  - [x] 3.1 Implementar `SlidingWindowBuffer` em `apps/backend/neurosign_backend/application/sliding_window.py`
    - Manter `dict[str, deque]` por `session_id` com `maxlen=window_size`
    - Retornar `np.ndarray` shape `(window_size, 84)` quando buffer cheio, senão `None`
    - Avançar `stride` frames após emissão descartando os mais antigos
    - _Requisitos: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [ ]* 3.2 Escrever teste de propriedade para isolamento de sessão (Propriedade 4)
    - **Propriedade 4: Isolamento de sessão no sliding window**
    - **Valida: Requisito 2.5**
  - [ ]* 3.3 Escrever teste de propriedade para emissão no momento correto (Propriedade 5)
    - **Propriedade 5: Sliding window emite janela no momento correto**
    - **Valida: Requisitos 3.1, 3.3, 3.5**
  - [ ]* 3.4 Escrever teste de propriedade para avanço correto após emissão (Propriedade 6)
    - **Propriedade 6: Sliding window avança corretamente após emissão**
    - **Valida: Requisito 3.4**

- [x] 4. Adaptador ONNX e caso de uso de inferência
  - [x] 4.1 Implementar `OnnxInferenceAdapter` em `apps/backend/neurosign_backend/adapters/onnx_adapter.py`
    - Carregar `InferenceSession` uma única vez no `__init__`
    - Executar `session.run` com input shape `(1, window_size, 84)`
    - Aplicar softmax e retornar Top-5 como `list[Prediction]` ordenada por score desc
    - Lançar `RuntimeError` descritivo se arquivo `.onnx` não existir
    - _Requisitos: 4.1, 4.2, 4.4, 4.5, 10.2_
  - [ ]* 4.2 Escrever teste de propriedade para Top-5 válido e ordenado (Propriedade 7)
    - **Propriedade 7: Inference Engine retorna Top-5 válido e ordenado**
    - **Valida: Requisito 4.2**
  - [ ]* 4.3 Escrever testes unitários para `OnnxInferenceAdapter`
    - Caso: carregamento único do modelo; caso: modelo não encontrado
    - _Requisitos: 4.4, 4.5_
  - [x] 4.4 Implementar `RunInferenceUseCase` em `apps/backend/neurosign_backend/application/run_inference.py`
    - Orquestrar `SessionPort.add_frame` → se janela: `InferencePort.predict` → retornar resultado
    - _Requisitos: 4.1, 10.1_

- [x] 5. WebSocket Handler e Prediction Broadcaster
  - [x] 5.1 Implementar `WebSocketSessionAdapter` em `apps/backend/neurosign_backend/adapters/ws_adapter.py`
    - Aceitar conexão em `/ws/{session_id}`
    - Loop: receber JSON → validar shape `(84,)` → `add_frame` → se janela: `predict` → enviar `PredictionMessage`
    - Tratar JSON inválido (fechar com código 1003), tensor com dimensão errada (descartar frame), desconexão abrupta (limpar buffer)
    - _Requisitos: 2.1, 2.4, 2.5, 5.1, 10.2_
  - [ ]* 5.2 Escrever teste de propriedade para serialização de predição (Propriedade 8)
    - **Propriedade 8: Serialização de predição contém todos os campos obrigatórios**
    - **Valida: Requisito 5.1**
  - [ ]* 5.3 Escrever testes unitários para `WebSocketHandler`
    - Caso: JSON inválido; caso: tensor com dimensão errada; caso: desconexão abrupta
    - _Requisitos: 2.4, 2.5_
  - [x] 5.4 Criar `apps/backend/neurosign_backend/main.py`
    - Instanciar `EnvConfig`, `OnnxInferenceAdapter`, `SlidingWindowBuffer`, registrar rota WebSocket
    - _Requisitos: 4.4, 4.5, 9.1, 9.3_

- [x] 6. Checkpoint — Backend
  - Garantir que todos os testes do backend passem. Verificar que o servidor inicializa corretamente com variáveis de ambiente definidas. Perguntar ao usuário se há dúvidas antes de continuar.

- [x] 7. Normalização de landmarks no frontend
  - [x] 7.1 Implementar `normalize` em `apps/frontend/src/lib/normalizer.ts`
    - Transladar subtraindo `landmark[0]` (pulso) de todos os pontos
    - Escalar dividindo pela distância euclidiana entre `landmark[0]` e `landmark[9]`
    - Retornar `Float32Array[84]`; se distância for zero, retornar zeros
    - _Requisitos: 1.2, 1.3, 1.4_
  - [ ]* 7.2 Escrever teste de propriedade para normalização (Propriedade 1)
    - **Propriedade 1: Normalização preserva invariantes geométricos**
    - **Valida: Requisitos 1.2, 1.3**
  - [ ]* 7.3 Escrever testes unitários para `normalizer.ts`
    - Caso normal, mão ausente (zeros), landmarks em posições extremas
    - _Requisitos: 1.2, 1.3, 1.4, 10.4_

- [x] 8. WebSocket Client e hook `useWebSocket`
  - [x] 8.1 Implementar `wsClient` em `apps/frontend/src/lib/wsClient.ts`
    - Manter conexão única persistente
    - Serializar tensor como `{ "frame": [...] }` em JSON
    - Reconexão com backoff exponencial: `delay = min(2^n * 1000, 30000)`
    - Expor `send(frame: Float32Array)` e `onMessage(handler)`
    - _Requisitos: 2.1, 2.2, 2.3, 2.4_
  - [ ]* 8.2 Escrever teste de propriedade para serialização de tensor (Propriedade 2)
    - **Propriedade 2: Serialização de tensor preserva valores**
    - **Valida: Requisito 2.1**
  - [ ]* 8.3 Escrever teste de propriedade para backoff exponencial (Propriedade 3)
    - **Propriedade 3: Backoff exponencial respeita limites**
    - **Valida: Requisito 2.3**
  - [ ]* 8.4 Escrever testes unitários para `wsClient.ts`
    - Caso: conexão única; caso: reconexão após queda; caso: rejeição de conexão
    - _Requisitos: 2.2, 2.3, 2.4_
  - [x] 8.5 Implementar hook `useWebSocket` em `apps/frontend/src/hooks/useWebSocket.ts`
    - Encapsular `wsClient`, expor `sendFrame` e `lastPrediction`
    - _Requisitos: 2.2, 2.3_

- [x] 9. MediaPipe e hook `useMediaPipe`
  - [x] 9.1 Implementar hook `useMediaPipe` em `apps/frontend/src/hooks/useMediaPipe.ts`
    - Inicializar `MediaPipe Hands` via WASM com `maxNumHands=2`
    - Processar cada frame do `HTMLVideoElement` e emitir `Float32Array[84]`
    - Se nenhuma mão detectada, emitir tensor de zeros
    - Processar a ≥ 15 FPS (usar `requestAnimationFrame`)
    - _Requisitos: 1.1, 1.4, 1.5_

- [x] 10. Componentes React e interface do usuário
  - [x] 10.1 Implementar `VideoFeed` em `apps/frontend/src/components/VideoFeed.tsx`
    - Renderizar `<video>` com stream da câmera
    - Solicitar permissão de câmera; exibir mensagem de erro se negada
    - _Requisitos: 1.1_
  - [x] 10.2 Implementar `PredictionDisplay` em `apps/frontend/src/components/PredictionDisplay.tsx`
    - Exibir `predictions[0].label` como tradução principal
    - Em modo expandido, listar Top-5 com barras de confiança
    - Atualizar estado com debounce ≤ 100ms via `useEffect`
    - Botão para habilitar síntese de voz via Web Speech API
    - _Requisitos: 5.2, 5.3, 5.4, 5.5_
  - [x] 10.3 Implementar `Controls` e montar `App.tsx`
    - Conectar `useMediaPipe` → `useWebSocket` → `PredictionDisplay`
    - Exibir indicador de reconexão e "aguardando sinal" conforme estados
    - _Requisitos: 2.3, 2.4, 5.2_
  - [ ]* 10.4 Escrever testes unitários para `PredictionDisplay`
    - Caso: exibição de tradução principal; caso: modo expandido; caso: síntese de voz
    - _Requisitos: 5.2, 5.3, 5.5_

- [x] 11. Checkpoint — Frontend
  - Garantir que todos os testes do frontend passem. Verificar que a aplicação renderiza corretamente e conecta ao backend. Perguntar ao usuário se há dúvidas antes de continuar.

- [x] 12. Pipeline de dados ML
  - [x] 12.1 Implementar `download.py` em `ml-lab/neurosign_ml/data/`
    - Baixar WLASL via `kaggle` API com credenciais de `KAGGLE_USERNAME` e `KAGGLE_KEY`
    - Lançar `ValueError` descritivo se credenciais ausentes
    - _Requisitos: 6.1_
  - [x] 12.2 Implementar `filter.py` — selecionar top-50 sinais por frequência de amostras
    - _Requisitos: 6.2_
  - [x] 12.3 Implementar `normalize.py` — mesma lógica de `normalizer.ts` em Python
    - _Requisitos: 6.3_
  - [ ]* 12.4 Escrever teste de propriedade para normalização Python (Propriedade 1 — validação cruzada)
    - **Propriedade 1: Normalização preserva invariantes geométricos**
    - **Valida: Requisitos 1.2, 1.3, 6.3**
  - [x] 12.5 Implementar `split.py` — dividir treino/val/teste sem vazamento de sinal entre conjuntos
    - Proporções configuráveis via parâmetros
    - _Requisitos: 6.4_
  - [x] 12.6 Implementar `dataset.py` — `torch.utils.data.Dataset` que carrega tensores serializados
    - _Requisitos: 6.5_
  - [ ]* 12.7 Escrever testes unitários para o pipeline de dados
    - Caso: filtragem top-50; caso: split sem vazamento
    - _Requisitos: 6.2, 6.4_

- [x] 13. Modelo BiLSTM com atenção e pipeline de treinamento
  - [x] 13.1 Implementar `BiLSTMAttention` em `ml-lab/neurosign_ml/models/bilstm_attention.py`
    - Encoder BiLSTM, mecanismo de atenção dot-product, classificador linear
    - _Requisitos: 7.1_
  - [x] 13.2 Implementar `Trainer` em `ml-lab/neurosign_ml/training/trainer.py`
    - Loop de treino com suporte a `device = mps | cuda | cpu`
    - Salvar top-3 checkpoints por `val_top1`
    - Logar métricas (loss, Top-1, Top-5) em CSV + `SummaryWriter`
    - Suporte a `--resume checkpoint.pt`
    - _Requisitos: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 14. Exportação ONNX, quantização e benchmark
  - [x] 14.1 Implementar `export_onnx.py` em `ml-lab/neurosign_ml/export/`
    - `torch.onnx.export` com opset 17
    - _Requisitos: 8.1_
  - [x] 14.2 Implementar `quantize.py` — `quantize_static` Float32 → Int8
    - _Requisitos: 8.2_
  - [x] 14.3 Implementar `benchmark.py` — medir P50/P95 em 100 amostras de teste
    - _Requisitos: 8.4, 4.3_
  - [x] 14.4 Implementar `validate.py` — comparar Top-1 Float32 vs Int8; falhar se delta > 2pp
    - _Requisitos: 8.5_
  - [ ]* 14.5 Escrever teste de propriedade para degradação de acurácia (Propriedade 10)
    - **Propriedade 10: Degradação de acurácia na quantização é aceitável**
    - **Valida: Requisito 8.5**
  - [ ]* 14.6 Escrever testes unitários para exportação
    - Caso: tamanho do arquivo ONNX < 10MB; caso: benchmark de latência
    - _Requisitos: 8.3, 8.4_

- [x] 15. Teste de integração do fluxo WebSocket completo
  - [x] 15.1 Implementar teste de integração em `apps/backend/tests/test_integration.py`
    - Usar `pytest-asyncio` + `httpx` com modelo stub de dimensões equivalentes
    - Fluxo: envio de tensor → sliding window → inferência → resposta JSON
    - _Requisitos: 10.5_

- [ ] 16. Infraestrutura Docker e documentação
  - [x] 16.1 Criar `apps/backend/Dockerfile` e `apps/frontend/Dockerfile`
    - _Requisitos: 9.1_
  - [ ] 16.2 Criar `docker-compose.yml` na raiz do monorepo
    - Serviços: `backend` e `frontend` com variáveis de ambiente documentadas
    - _Requisitos: 9.1, 9.3_
  - [-] 16.3 Criar `README.md` com diagrama de fluxo de dados, instruções de setup e link para demonstrativo
    - _Requisitos: 9.4_

- [ ] 17. Checkpoint final — Garantir que todos os testes passem
  - Executar suite completa de testes (backend, frontend, ml-lab). Verificar que `docker-compose up` sobe todos os serviços sem erros. Perguntar ao usuário se há dúvidas antes de concluir.

## Notas

- Tarefas marcadas com `*` são opcionais e podem ser puladas para um MVP mais rápido
- Cada tarefa referencia requisitos específicos para rastreabilidade
- Os checkpoints garantem validação incremental a cada frente de trabalho
- Testes de propriedade validam invariantes universais; testes unitários validam exemplos e casos de borda
- O modelo ONNX deve ser gerado pelo ML Lab antes de executar os testes de integração do backend
