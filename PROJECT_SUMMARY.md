# Digital Phytomer: Documentação Científica e Técnica do Projeto

Este documento fornece um detalhamento completo da arquitetura, formulação matemática, hipóteses de pesquisa e estrutura de software do projeto **Digital Phytomer**. Ele foi projetado para que outro modelo de linguagem (LLM) compreenda o propósito e possa reproduzir ou estender o sistema de forma autônoma.

---

## 1. Visão Geral e Propósito

O **Digital Phytomer** é um framework de simulação de enxames de microagentes descentralizados que modela dinâmicas de auto-organização, divisão de trabalho e adaptabilidade topológica. Ele é inspirado na botânica, especificamente na organização de **fitômeros** (metâmeros estruturais compostos por nó, entrenó e gema axilar) e na distribuição de recursos pelo sistema vascular das plantas.

### Problema que Resolve
Sistemas centralizados com LLMs massivos são:
1. **Frágeis:** Uma única falha de conexão ou esgotamento de limite de taxa derruba o sistema.
2. **Pesados Computacionalmente:** Processar janelas gigantescas de contexto para tarefas repetitivas é energeticamente ineficiente.
3. **Rígidos:** Estruturas de roteamento estático não lidam bem com desvios repentinos de domínio (*paradigm shifts*) ou perda física de nós.

### Solução Proposta
Uma população de microagentes leves ($N=20$) alimentados por um modelo local menor (`qwen2.5:0.5b`). O enxame não possui controle central. A inteligência e a divisão do trabalho emergem por meio de **jogos de leilão competitivo**, **delegações entre vizinhos locais**, **mecanismos de aprendizado somático** e um **processador de gramática L-System** que regenera nós mortos (depletados de energia) readequando suas conexões com base em gradientes locais e globais de recursos.

---

## 2. Hipóteses Científicas de Pesquisa

O sistema foi construído para testar três hipóteses centrais:

1. **Hipótese da Especialização Emergente (Divisão de Trabalho):** Um leilão descentralizado baseado na similaridade semântica da memória somática local do agente, acrescido de uma taxa de monopólio progressiva ($\tau_{\text{monopoly}}$), estimula o enxame a alcançar um alto índice de diferenciação funcional (FDI $\to 1.0$) sem coordenação centralizada.
2. **Hipótese da Plasticidade e Robustez Topológica:** Em cenários de lesão celular simulada (abscisão/morte súbita dos nós mais competentes), a regeneração ativa baseada em L-Systems, avaliando o gradiente de recursos locais em relação à média global, acelera a recuperação do enxame comparado a topologias de roteamento estático e fixo.
3. **Hipótese do Custo de Compressão Semântica:** A destilação recorrente de logs de tentativas locais de curto prazo (memória episódica) em lições de longo prazo (memória somática vetorial) reduz drasticamente a contagem de tokens processados ($>60\%$ de economia de contexto) sem degradar a acurácia final.

---

## 3. Formulação Matemática dos Modelos Coletivos

### 3.1. Memória Somática e Difusão RAG-VM (Retrieval-Augmented Generation Vector Manifold)
A memória somática de cada microagente é uma coleção de vetores no espaço $\mathbb{R}^{768}$ gerados via `nomic-embed-text`. Para modelar a intercomunicabilidade implícita e a proximidade semântica dos problemas resolvidos, o sistema aplica um algoritmo de **Difusão Espectral no Grafo (Normalized Laplacian Exponentiation)** offline.

1. **Matriz de Adjacência ($W$):** Computa a proximidade de cosseno atenuada entre os vetores de memória do agente usando largura de banda ($\sigma$) e atenuação de borda ($\kappa$):
   $$W_{ij} = e^{-\frac{\|v_i - v_j\|^2}{\sigma^2}} \cdot e^{-\kappa \|v_i - v_j\|^2}$$
2. **Laplaciano Normalizado ($L$):**
   $$L = I - D^{-1/2} W D^{-1/2}$$
   Onde $D$ é a matriz de graus diagonal ($D_{ii} = \sum_j W_{ij}$).
3. **Difusão Espectral (Exponencial da Matriz):** Os embeddings originais $\Phi$ são propagados pelo grafo de memória:
   $$\Phi_{\alpha} = e^{-\alpha L} \Phi$$
   Esta propagação é computada via decomposição espectral do Laplaciano: $e^{-\alpha L} = V e^{-\alpha \Lambda} V^T$.
4. **Recuperação por Duplo Índice (Dual-Index Retrieval):** A busca semântica do agente combina a similaridade bruta e a similaridade difusa (para capturar conexões conceituais de segunda ordem):
   $$\text{Sim}_{\text{dual}} = (1 - \lambda) \cdot \text{Sim}_{\text{raw}} + \lambda \cdot \text{Sim}_{\text{diffused}} \quad (\text{com } \lambda = 0.5)$$
5. **Decaimento e Relevância:** A pontuação final é atenuada pela relevância dinâmica do vetor (reforçada sob sucesso, degradada sob falha e sujeita a decaimento temporal passivo) e incrementada por um bônus de recência:
   $$\text{Score} = \text{Sim}_{\text{dual}} \cdot \text{Relevance} \cdot (1 + \beta e^{-\frac{\Delta t}{\tau_r}})$$

### 3.2. Mecanismo de Leilão e Taxa de Monopólio
Quando uma tarefa chega ao enxame, os nós ativos disputam a sua execução através de lances de competência:
$$\text{Bid}_i = \text{Competence}_i \cdot 0.6 + (1.0 - \text{Monopoly Tax}_i) \cdot 0.4 + \epsilon$$

Onde:
* $\text{Competence}_i$: Similaridade da tarefa com a memória de sucesso de $i$ via consulta RAG-VM.
* $\epsilon$: Ruído gaussiano que simula flutuações físicas ou de canais de comunicação ($\epsilon \sim \mathcal{N}(0, \text{noise\_level})$).
* $\text{Monopoly Tax}_i$: Imposto punitivo aplicado a nós superespecializados para evitar o bloqueio e gargalo de execução por fadiga do agente:
  $$\text{Monopoly Tax}_i = \min\left(0.4, \text{Tasks Solved}_i \cdot \tau_{\text{monopoly}}\right)$$

### 3.3. Dinâmica de Atualização de Confiança (Trust Network)
As relações topológicas não são físicas, mas lógicas, representadas por tabelas de confiança mútua entre os agentes.
* **Sob Sucesso de Delegação:** O agente que delegou aumenta a confiança no ajudante:
  $$T_{ij} \leftarrow \min(1.0, T_{ij} + \Delta_{\text{success}})$$
* **Sob Falha de Delegação:** O agente que delegou pune a confiança no ajudante:
  $$T_{ij} \leftarrow \max(0.0, T_{ij} - \Delta_{\text{failure}})$$
* **Reciprocidade (Feedback Bidirecional):** Se a tarefa é resolvida com sucesso, a confiança de $j$ em $i$ também recebe um bônus residual de cooperação:
  $$T_{ji} \leftarrow \min(1.0, T_{ji} + 0.01)$$

---

## 4. O Sistema Gramatical L-System (Regeneração Topológica)

Quando os recursos energéticos de um microagente chegam a zero, o nó sofre **abscisão** (morte lógica). A regeneração de um novo nó no seu lugar lógico não é imediata e segue regras estritas de uma gramática generativa parametrizada (L-System):

### Alfabeto de Símbolos
* `A`: Axioma inicial de bud adormecido.
* `W(n)`: Contador de passos de espera (tempo de maturação celular).
* `M`: Estado meristemático (pronto para divisão e alongamento).
* `T`: Transição de avaliação de gradiente ambiental.
* `C(k)`: Conexão ativa com $k$ vizinhos.

### Regras de Produção Gramatical
1. **Regra de Inicialização:**
   $$A \to W(5) M$$
2. **Regra de Maturação (Passagem de Tempo):**
   $$W(n) \to W(n-1) \quad (\text{para } n > 1)$$
3. **Regra de Ativação Meristemática (Fim da Espera):**
   $$W(1) M \to T C(3)$$
4. **Regra de Avaliação de Gradiente (Topologia Dinâmica):**
   O nó avalia a saúde local (média dos recursos de seus 3 vizinhos originais no momento da morte, $\mu_{\text{local}}$) em relação à saúde global do enxame ($\mu_{\text{global}}$):
   * **Regeneração Local (Regra 5):** Se $\mu_{\text{local}} > \mu_{\text{global}}$, o nó regenera reconectando-se aos seus 3 vizinhos originais.
   * **Regeneração por Plasticidade Topológica (Regra 6):** Se $\mu_{\text{local}} \le \mu_{\text{global}}$, o ambiente local está degradado. O nó estende suas "raízes" para fora, conectando-se aos 3 agentes mais ricos em recursos do enxame como um todo.

---

## 5. Módulos, Classes e Funções Cruciais

O ecossistema é composto pelos seguintes arquivos em Python:

### 5.1. `micro_agent.py`
Representa a unidade celular do enxame (`MicroAgent`).
* **Atributos:** `agent_id` (ex: `Cell-001`), `role` (papel dinâmico), `system_prompt`, `resource` (energia), `failures_count` (contador de falhas consecutivas local).
* **`solve(problem)`:** Executa a inferência local. Combina o contexto semântico retornado do RAG-VM e o histórico episódico recente de falhas em um prompt unificado, chamando o cliente LLM com temperatura variável (0.3 estável, 0.8 sob mutação estocástica).

### 5.2. `vector_store.py`
Contém a classe `SomaticVectorStore` responsável pelo armazenamento semântico.
* **`add_document(text, embedding)`:** Adiciona novos padrões de sucesso e executa a verificação de interferência de memória (se a similaridade com um vetor antigo for $>0.8$, reduz a relevância do vetor anterior para evitar saturação). Executa o RAG-VM spectral diffusion.
* **`query(query_embedding)`:** Executa a busca em duplo índice (cosseno sobre o embedding bruto e o embedding espectralmente difundido).
* **`apply_temporal_decay()`:** Realiza o decaimento exponencial passivo com ruído gaussiano nas memórias que não participaram das últimas tarefas.

### 5.3. `cognitive_memory.py`
Implementa a classe `HierarchicalMemory`.
* **`add_episode()`:** Armazena logs imediatos de tentativas.
* **`compress_to_semantic(client)`:** Função vital para a economia de contexto. Quando a memória episódica local atinge o teto (`capacity=3`), chama o LLM para destilar as razões de falha e criar uma regra semântica de "lição aprendida" com menos de 80 palavras. Limpa os logs episódicos antigos e armazena a lição gerada no banco vetorial.

### 5.4. `lsystem_regenerator.py`
Gerencia a evolução gramatical das células mortas.
* **`step(agents, active_ids)`:** Executa as regras de transição gramaticais e calcula a média de recursos locais vs. globais, retornando a nova matriz de conexões para reinstanciar a célula regenerada.

### 5.5. `cognitive_verifier.py`
Garante a segurança e a validação lógica das respostas.
* **`RestrictedPythonExecutor`:** Varre a Árvore de Sintaxe Abstrata (AST) do código gerado por LLMs antes de rodá-lo, bloqueando chamadas perigosas (`os`, `subprocess`, `socket`, `open()`).
* **`execute()`:** Executa o código em um subprocesso Linux limitado através do módulo POSIX `resource` (teto rígido de 3 segundos de CPU e 128MB de RAM) para impedir loops infinitos e vazamentos de memória na máquina física.

### 5.6. `ollama_client.py`
Gerencia a conexão com o servidor local do Ollama (`http://localhost:11434`).
* **Características:** Armazena as estatísticas globais de telemetria de tokens, implementa cache em disco (`results/llm_cache.json`) de forma determinística por hash de prompt/seed, e define tempos limites de rede rígidos (`timeout`) para evitar travamentos indefinidos da simulação.

---

## 6. Pipeline de Execução (`run_all.py`)

A automação dos experimentos é encadeada por [run_all.py](file:///home/destritux/Projetos/Digital%20Phytomer/run_all.py), que executa sequencialmente em modo não-bufferizado (`-u`):

1. **`unittest test_lsystem_regenerator.py`:** Executa a validação das regras gramaticais e as decisões de vizinhança local/global.
2. **`simulate_emergent_cognition.py`:** Roda os experimentos multissemente (5 sementes, 5 épocas, 100 tarefas compostas por Matemática, Cibersegurança, Robótica e Engenharia Reversa) comparando o Phytomer Emergente com os baselines e gerando gráficos em `results/`.
3. **`generate_final_docx.py`:** Compila os gráficos e formata o artigo científico final baseado nas diretrizes do IEEE/ACM.

---

## 7. Como o LLM Deve Monitorar e Iterar neste Projeto

Para interagir ou modificar este projeto, siga estas diretrizes:

1. **Evitar Redirecionamentos de Output:** Nunca use manipuladores de contexto que fechem ou capturem o `sys.stdout` global do Python de forma destrutiva (como o antigo `SuppressStdout`), pois threads em segundo plano (especialmente do OpenBLAS/Numpy e urllib3) entrarão em conflito, travando o interpretador.
2. **Verificar os Timeouts de Rede:** Certifique-se de que qualquer chamada HTTP ao Ollama possua o parâmetro `timeout` especificado.
3. **Gerenciar o Cache:** O arquivo [results/llm_cache.json](file:///home/destritux/Projetos/Digital%20Phytomer/results/llm_cache.json) é o repositório de telemetria. Se alterar as tarefas em `simulate_emergent_cognition.py`, as novas chamadas gerarão novos hashes de cache. Certifique-se de que o modelo local do Ollama (`qwen2.5:0.5b`) esteja ativo no sistema para evitar o uso exclusivo dos fallbacks determinísticos.
