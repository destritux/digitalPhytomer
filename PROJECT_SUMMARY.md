# Digital Phytomer: Documentação Científica e Técnica do Projeto (Versão 2.0)

Este documento fornece um detalhamento completo da arquitetura, formulação matemática, hipóteses de pesquisa e estrutura de software do projeto **Digital Phytomer**. Ele foi projetado para servir como uma referência técnica exaustiva para desenvolvedores e outros agentes inteligentes compreenderem e estenderem o sistema de forma autônoma.

---

## 1. Visão Geral e Propósito

O **Digital Phytomer** é um framework de simulação de enxames de microagentes descentralizados que modela dinâmicas de auto-organização, divisão de trabalho e adaptabilidade topológica. Ele é inspirado na botânica, especificamente na organização de **fitômeros** (metâmeros estruturais compostos por nó, entrenó e gema axilar) e na distribuição de recursos pelo sistema vascular das plantas (Sink-Source gradient).

### Problema que Resolve
Sistemas centralizados com LLMs massivos são:
1. **Frágeis:** Uma única falha de conexão ou esgotamento de limite de taxa derruba o sistema.
2. **Pesados Computacionalmente:** Processar janelas gigantescas de contexto para tarefas repetitivas é energeticamente ineficiente.
3. **Rígidos:** Estruturas de roteamento estático não lidam bem com desvios repentinos de domínio (*paradigm shifts*) ou perda física de nós.

### Solução Proposta
Uma população de microagentes leves ($N=20$) alimentados por um modelo local menor (`qwen2.5:0.5b`). O enxame não possui controle central. A inteligência e a divisão do trabalho emergem por meio de **jogos de leilão competitivo**, **delegações entre vizinhos locais**, **mecanismos de aprendizado somático**, **memória associativa em grafo** e um **processador de gramática L-System** que regenera nós mortos (depletados de energia) readequando suas conexões com base em gradientes locais e globais de recursos.

---

## 2. Hipóteses Científicas de Pesquisa

O sistema foi construído para testar quatro hipóteses centrais:

1. **Hipótese da Especialização Emergente (Divisão de Trabalho):** Um leilão descentralizado baseado na similaridade semântica da memória somática local do agente, acrescido de uma taxa de monopólio progressiva ($\tau_{\text{monopoly}}$), estimula o enxame a alcançar um alto índice de diferenciação funcional (FDI $\to 1.0$) sem coordenação centralizada.
2. **Hipótese da Plasticidade e Robustez Topológica:** Em cenários de lesão celular simulada (abscisão/morte súbita dos nós mais competentes), a regeneração ativa baseada em L-Systems, avaliando o gradiente de recursos locais em relação à média global, acelera a recuperação do enxame comparado a topologias de roteamento estático e fixo.
3. **Hipótese do Custo de Compressão Semântica:** A destilação recorrente de logs de tentativas locais de curto prazo (memória episódica) em lições de longo prazo (memória somática vetorial) reduz drasticamente a contagem de tokens processados ($>60\%$ de economia de contexto) sem degradar a acurácia final.
4. **Hipótese da Resiliência Ecológica Baseada em Grafo e MARL:** A integração de uma rede micorrízica global de reflexões consolidada (GAAMA/SAGE), combinada com custos metabólicos de transmissão (Gated Communication) e aprendizado cooperativo por destilação federada (MEPD-PPO), previne o colapso ecológico em cascata provocado pela saturação de Carga Cognitiva e Etileno Virtual.

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

### 3.2. Mecanismo de Leilão com Cross-Attention Vetorial
Quando uma tarefa chega ao enxame, os nós ativos disputam a sua execução através de lances de competência. O controlador de árvore (Tree Controller) não calcula a pressão vascular local baseado em médias cegas. É calculado um peso de atenção vetorial $\alpha_{ij}$ entre os agentes da vizinhança de confiança (`local_peers`) e o problema atual ($q\_emb$):
$$\alpha_{ij} = \text{Sim}(q\_emb, \text{vector\_store}_j)$$

O bid final de cada microagente é modelado como:
$$\text{Bid}_i = \text{Competence}_i \cdot 0.6 + (1.0 - \text{Monopoly Tax}_i) \cdot 0.4 + \epsilon$$

Onde:
* $\text{Competence}_i$: Similaridade da tarefa com a memória de sucesso de $i$ via consulta RAG-VM.
* $\epsilon$: Ruído gaussiano que simula flutuações físicas ($\epsilon \sim \mathcal{N}(0, \text{noise\_level})$).
* $\text{Monopoly Tax}_i$: Imposto punitivo aplicado a nós superespecializados para evitar gargalo de execução por fadiga do agente:
  $$\text{Monopoly Tax}_i = \min\left(0.4, \text{Tasks Solved}_i \cdot \tau_{\text{monopoly}}\right)$$

### 3.3. Silêncio Seletivo (Gated Communication)
Antes de acionar a busca micorrízica P2P ou global, o agente avalia sua incerteza local:
$$\text{Incerteza} = 1.0 - \text{best\_local\_memory\_score}$$

A comunicação externa só é autorizada se o valor esperado superar o custo metabólico (virtual ATP):
$$(\text{Incerteza} \cdot \text{REWARD}) > \text{COMMUNICATION\_COST} \quad (\text{onde } \text{COMMUNICATION\_COST} = 2)$$

### 3.4. Dinâmica de Confiança (Trust Network) e Decaimento
As relações topológicas lógicas são representadas por tabelas de confiança mútua ($T_{ij}$).
* **Sob Sucesso de Delegação:** $T_{ij} \leftarrow \min(1.0, T_{ij} + \Delta_{\text{success}} \cdot 3)$ (Spike Hebbiano triplicado se o conselho destilado for útil).
* **Sob Falha de Delegação:** $T_{ij} \leftarrow \max(0.0, T_{ij} - \Delta_{\text{failure}})$
* **Decaimento Passivo Temporal:** Para mitigar dependências permanentes, a matriz de confiança decai passivamente a cada passo da simulação:
  $$T_{ij} \leftarrow T_{ij} \cdot 0.99$$

---

## 4. O Sistema Gramatical L-System (Regeneração Topológica)

Quando os recursos energéticos de um microagente chegam a zero, o nó sofre **abscisão** (morte lógica). A regeneração de um novo nó segue regras de uma gramática generativa parametrizada (L-System):

### Alfabeto de Símbolos
* `A`: Axioma inicial de bud adormecido.
* `W(n)`: Contador de passos de espera (tempo de maturação celular).
* `M`: Estado meristemático (pronto para divisão e alongamento).
* `T`: Transição de avaliação de gradiente ambiental.
* `C(k)`: Conexão ativa com $k$ vizinhos.

### Regras de Produção Gramatical
1. **Regra de Inicialização:** $A \to W(5) M$
2. **Regra de Maturação (Passagem de Tempo):** $W(n) \to W(n-1) \quad (\text{para } n > 1)$
3. **Regra de Ativação Meristemática (Fim da Espera):** $W(1) M \to T C(3)$
4. **Regra de Avaliação de Gradiente (Topologia Dinâmica):**
   * **Regeneração Local (Regra 5):** Se $\mu_{\text{local}} > \mu_{\text{global}}$, o nó regenera reconectando-se aos seus 3 vizinhos originais.
   * **Regeneração por Plasticidade Topológica (Regra 6):** Se $\mu_{\text{local}} \le \mu_{\text{global}}$, o nó estende suas raízes, conectando-se aos 3 agentes mais ricos do enxame.

---

## 5. Estrutura do Código: Módulos, Classes e Métodos

O ecossistema é modularizado em componentes especialistas que interagem de forma síncrona e assíncrona.

### 5.1. `vector_store.py`
Gerencia a persistência semântica e a difusão espectral local de cada agente, além da rede global compartilhada.

#### Classe `SomaticVectorStore`
* `add_document(text, embedding, metadata)`: Insere um novo fato semântico no banco local. Implementa verificação de interferência de memória (se similaridade com documento existente for $>0.8$, a relevância do existente é multiplicada por $0.95$). Limita a capacidade do agente a 10 documentos, eliminando os de menor relevância.
* `_update_diffused_embeddings()`: Reconstrói o Laplaciano Normalizado ($L$) a partir da proximidade de cosseno atenuada entre os vetores de memória do agente e aplica a difusão spectral $e^{-\alpha L} \Phi$.
* `query(query_embedding, limit, min_similarity)`: Consulta o banco de dados local com indexação dupla. O escore é composto pela média ponderada das similaridades bruta e difusa, escalado pela relevância do vetor e incrementado por um bônus de recência exponencial temporal.
* `apply_feedback(active_texts, success)`: Reforça o escore de relevância dos vetores usados em caso de sucesso ($+0.1$) ou reduz em caso de falha ($*0.85$).
* `apply_temporal_decay(active_texts)`: Aplica decaimento passivo modulado pela força de memória $k(t)$ (calculada via sigmoide com base no custo de estresse dinâmico local) e adiciona um ruído gaussiano.
* `prune_low_relevance_vectors(threshold=0.25)`: Exclui permanentemente memórias cuja relevância caia abaixo do limite.

#### Classe `GlobalMemoryNetwork` (e instância `global_memory_network`)
* `push_reflection(text, embedding, source_facts)`: Armazena Reflexões de Ordem Superior consolidadas globalmente.
* `query(query_embedding, limit, min_similarity)`: Busca reflexões globais que correspondam semânticamente à consulta e retorna as reflexões encontradas juntamente com os fatos atômicos originais (`source_facts`) que as geraram para fornecer fundamentação contextual sólida ao prompt.

---

### 5.2. `cognitive_memory.py`
Implementa o subsistema de Memória Hierárquica Associativa inspirado nos modelos GAAMA e SAGE.

#### Classe `HierarchicalMemory`
* `add_episode(prompt, response, evaluation_meta, client)`: Salva o log episódico cru (tentativa de execução + `stderr` retornado). Se a fila atingir o limite (`capacity=3`), aciona automaticamente a compressão semântica.
* `compress_to_semantic(client)`: Extrai fatos atômicos objetivos das tentativas malsucedidas por meio do LLM estruturado com formato JSON. Salva estes novos fatos locais no `SomaticVectorStore` e limpa os logs antigos.
* `synthesize_reflections(client)`: Identifica grupos de 3 ou mais fatos locais com similaridade semântica superior a $0.70$, chama o LLM para destilar uma *Reflexão de Ordem Superior* (heurística geral de menos de 40 palavras) e a envia para o `GlobalMemoryNetwork` compartilhado. Prune os 3 fatos de origem da memória local.
* `distill_expert_policy(query_text, client)`: Gera uma heurística destilada concisa ($<30$ palavras) baseada nas memórias somáticas do especialista para orientar o prompt de um agente estudante sob estresse cognitivo (Policy Distillation).
* `retrieve_context(query_text, client)`: Busca e formata o conhecimento semântico local pertinente ao prompt da tarefa em andamento.

---

### 5.3. `micro_agent.py`
Modela a unidade celular (`MicroAgent`) do fitômero computacional.

#### Classe `MicroAgent`
* `solve(problem, client)`: Função principal que coordena a inferência local. Sob alto estresse dinâmico ($>1.0$ derivado de falhas consecutivas locais e Carga Cognitiva), o agente toma uma decisão alostática: paga um custo metabólico de $-2$ recursos e executa uma consulta à rede micorrízica global (`GlobalMemoryNetwork`), incorporando os fatos de base da reflexão em seu prompt.
* Outros atributos geridos incluem `resource` (que decai sob falha e leilão) e `epigenetic_stress` (que escala a taxa de decaimento do etileno dinâmico).

---

### 5.4. `simulate_emergent_cognition.py`
Encapsula o motor de simulação, a governança coletiva e a lógica de atribuição de crédito.

#### Funções Principais
* `run_swarm(...)`: Inicializa a população de microagentes, constrói a vizinhança lógica (Rede de Confiança) e executa o ciclo de iterações de tarefas. Implementa o protocolo de atribuição de crédito baseados em MEPD-PPO:
  * Se o auxílio de um especialista (`helper_id`) via destilação de política federada resultar em sucesso: o especialista recupera seu custo de síntese ($-3$ recursos) e recebe um prêmio de $+200\%$ da recompensa padrão. O estudante absorve a diretriz heurística diretamente em sua memória somática local como um fato atômico e recebe $50\%$ da recompensa.
* `run_statistical_experiment(...)`: Executa varreduras multissementes (isolando cache de LLM) para compilar trajetórias de acurácia, FDI e dominância.
* `perform_statistical_tests(results_dict)`: Executa testes Wilcoxon/Mann-Whitney de relevância estatística, aplicando correção Holm-Bonferroni sobre os p-values compilados.
* `plot_specialization_heatmaps(...)`: Renderiza matrizes de alocação de tarefas para atestar o índice de diferenciação do enxame.

---

### 5.5. `lsystem_regenerator.py`
Gerencia a dinâmica de modelagem botânica da topologia de rede.

#### Classe `LSystemRegenerator`
* `step(agents, active_ids)`: Processa a passagem do tempo biológico e avalia as regras de L-System nos nós em estado meristemático ou de abscisão.
* `evaluate_gradients(...)`: Computa a diferença entre a saúde local da vizinhança celular e a saúde sistêmica global para ditar se o novo fitômero restabelecerá as conexões locais anteriores (Regeneração Estável) ou buscará novas adjacências dinâmicas com os nós mais ricos do ecossistema (Plasticidade Topológica).

---

### 5.6. `cognitive_verifier.py`
Atua como o sistema imunológico/verificador POSIX que valida as saídas das tarefas sem expor a integridade da máquina hospedeira.

#### Métodos de Segurança
* `RestrictedPythonExecutor.visit_Call(...)`: Filtra via AST importações ou execuções perigosas.
* `execute(...)`: Controla tempo limite e consumo de recursos de memória em tempo de execução via recursos nativos do kernel Linux (`resource.setrlimit`).

---

### 5.7. `ollama_client.py`
Interface unificada de comunicação com o servidor de inferência.

#### Classe `OllamaClient`
* `generate(prompt, system_prompt, json_format, temperature)`: Abstrai a chamada ao modelo local, suportando restrições de formato JSON estruturado por gramática do Ollama e gerindo o banco de cache determinístico em disco para acelerar testes recorrentes.

---

## 6. Pipeline de Execução (`run_all.py`)

A execução automatizada de todo o ecossistema é encadeada recursivamente pelo script [run_all.py](file:///home/destritux/Projetos/Digital%20Phytomer/run_all.py):
1. Executa os testes unitários (`test_lsystem_regenerator.py`).
2. Roda a suite de simulações multissemente completas com baselines e ablações (`simulate_emergent_cognition.py`).
3. Executa o gerador de relatórios e artigos científicos (`generate_final_docx.py`), formatando todos os gráficos gerados e métricas estatísticas em um documento Word IEEE-compliant.

---

## 7. Diretrizes para Desenvolvimento e Monitoramento

1. **Persistência de Memória:** O arquivo [results/llm_cache.json](file:///home/destritux/Projetos/Digital%20Phytomer/results/llm_cache.json) acelera as execuções repetidas. Qualquer alteração nos prompts gerará novos hashes que exigirão inferência em tempo real. Garanta que o serviço Ollama esteja ativo na máquina local.
2. **Saída Não-Capturada:** Não altere ou crie manipuladores que redirecionem `sys.stdout` de forma agressiva. Processos filhos e compiladores C subjacentes (numpy/OpenBLAS) causam deadlocks de thread se o descritor padrão de escrita estiver inacessível.
