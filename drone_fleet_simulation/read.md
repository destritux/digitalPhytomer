Este projeto implementa um **simulador bioinspirado de enxame de drones cognitivos** chamado **Digital Phytomer**. Ele combina conceitos de biologia vegetal (árvores, micorrizas, fluxo vascular) com inteligência artificial distribuída para modelar a resolução colaborativa de tarefas de navegação.

## O que ele faz, em resumo:

- Simula um **enxame de drones** (de 1 a 5), onde cada drone possui vários **sensores** (câmera, LiDAR, térmico, etc.) que atuam como **microagentes autônomos**.
- Cada sensor tenta resolver tarefas de navegação (ex.: desviar de obstáculo, escolher rota, interpretar dados de sensores) usando um **modelo de linguagem local** (Ollama) ou um **emulador probabilístico**.
- Implementa uma **arquitetura de três camadas**:
  - **TreeController (TC)** – Gerencia os sensores de um drone: alocação de tarefas (bidding), delegação entre sensores, ajuste de confiança (trust), consumo/recuperação de energia (“fluxo vascular”) e **memória somática** (armazenamento vetorial de lições aprendidas).
  - **ForestController (FC)** – Coordena o enxame inteiro, realizando **sincronização micorrízica**: compartilha memórias semânticas entre os drones, simulando troca de nutrientes/informação via rede de fungos.
- Inclui **mecanismos homeostáticos**:
  - Energia (recursos) de cada sensor e saúde geral do drone.
  - Se um sensor falha repetidamente ou fica sem energia, ele sofre **poda** (remoção definitiva).
  - Se um drone fica sem saúde, ele **cai** e para de operar.
  - O sistema também calcula **carga alostática** (custo metabólico acumulado) e **índice de especialização FDI** (desvio padrão das taxas de acerto entre drones).
- Oferece um **dashboard web interativo** (`index.html`) onde o usuário pode:
  - Configurar número de drones/sensores, velocidade da simulação, ativar/desativar uso real de LLM (Ollama).
  - Iniciar/pausar a simulação.
  - Visualizar em tempo real a topologia de cada drone, estado dos sensores (ativo, alerta, simbiose, podado, falha), nível de energia e saúde.
  - Alternar manualmente sensores entre ligado/desligado.
  - Acompanhar gráficos de acurácia do enxame, especialização FDI e carga alostática.
  - Ler logs do sistema e uma explicação botânica paralela do comportamento atual.

## Principais tecnologias usadas:

- **Backend**: Python com `http.server`, `socketserver`, `threading`, `numpy`, `json`.
- **LLM local**: Integração com **Ollama** (modelo padrão `qwen2.5:0.5b`) – pode ser desligada para simulação puramente heurística.
- **Memória vetorial**: Armazenamento simples com normalização, similaridade por cosseno, relevância dinâmica (reforço/penalização) e decaimento temporal.
- **Frontend**: HTML/CSS/JS puro, gráficos SVG dinâmicos, comunicação assíncrona com a API REST (`/api/status`, `/api/start`, `/api/stop`, `/api/config`, `/api/toggle_sensor`).

## Para que serve?

- **Estudo de algoritmos bioinspirados** para enxames autônomos.
- **Demonstração de arquiteturas cognitivas** com memória distribuída, aprendizado contínuo e adaptação por confiança.
- **Simulação de falhas** e recuperação em sistemas multiagente.
- **Benchmark** de uso de LLMs pequenos (embarcados) em tarefas de decisão de baixo custo computacional.

Em suma, é um **laboratório experimental** que une biologia vegetal, computação evolutiva e IA generativa para criar um sistema resiliente, auto-organizado e visualmente interativo.
