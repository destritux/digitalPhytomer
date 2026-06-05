# Proposta de Revisão Textual e Teórica — Digital Phytomer

Este documento apresenta blocos de texto prontos para substituição e integração no manuscrito principal do artigo **Digital Phytomer**, visando atender às exigências de rigor conceitual, neutralidade linguística e formalismo metodológico feitas pelos revisores.

---

## 1. Formulação da Hipótese Operacional Explícita

> [!NOTE]
> **Onde inserir:** No final da Introdução ou no início da seção de Metodologia, para posicionar claramente o escopo do experimento de forma falsificável.

### Texto Sugerido (Português)
Para avaliar rigorosamente as propriedades organizacionais da arquitetura proposta, formulamos a seguinte hipótese operacional de pesquisa:

$$\mathcal{H}_1: \text{FDI}_{\text{Phytomer}} > \text{FDI}_{\text{Baselines}} \quad \text{e} \quad \text{Switching Rate}_{\text{Phytomer}} < \text{Switching Rate}_{\text{Baselines}}$$

> "Arquiteturas multiagente descentralizadas baseadas em regulação local de confiança e memória somática descentralizada apresentam um aumento estatisticamente significativo ($p < 0.05$, corrigido por Holm-Bonferroni) no índice de especialização funcional (FDI) e uma redução na taxa de alternância temporal de tarefas (Switching Rate) por agente em comparação com baselines com roteamento puramente estocástico (Random Routing), controle centralizado e pipelines fixos de atribuição de papéis, mantendo estabilidade operacional sob quebra de paradigma."

### Text Suggestion (English)
To evaluate the organizational properties of the proposed architecture, we formulate the following operational research hypothesis:

$$\mathcal{H}_1: \text{FDI}_{\text{Phytomer}} > \text{FDI}_{\text{Baselines}} \quad \text{and} \quad \text{Switching Rate}_{\text{Phytomer}} < \text{Switching Rate}_{\text{Baselines}}$$

> "Decentralized multi-agent architectures regulated by local trust dynamics and somatic memory exhibit a statistically significant ($p < 0.05$, adjusted via Holm-Bonferroni) increase in functional specialization (FDI) and a lower consecutive task alternation rate (Switching Rate) per agent compared to classic baselines (Random Routing, Centralized Planner, and Fixed Role Assignment) while maintaining operational accuracy under paradigm shifts."

---

## 2. Tabela de Mitigação Linguística (Expurgo de Teleologia)

Para blindar o paper contra críticas de antropomorfismo e teleologia (atribuir "vontade" ou "intencionalidade" ao enxame), substituímos construções finalistas por terminologia estatística e observacional.

| Termo Original / Teleológico | Crítica do Revisor | Substituição Proposta / Neutra | Justificativa Científica |
| :--- | :--- | :--- | :--- |
| "O enxame decide..." | Atribui intencionalidade ou consciência coletiva subjetiva. | "O sistema converge para..." ou "O estado do enxame estabiliza em..." | Define o fenômeno como um atrator dinâmico, não uma decisão consciente. |
| "Os agentes aprenderam a colaborar..." | "Aprender" sugere treino de pesos neurais no nível do enxame. | "Emergiu um padrão cooperativo via calibração local de trustscores..." | Mantém o foco no mecanismo matemático de alocação de confiança. |
| "A floresta se adaptou com o objetivo de..." | Sugere finalismo teleológico (ação voltada a fins futuros). | "A arquitetura auto-organizou-se sob a restrição metabólica..." | Explica a adaptação como uma resposta reativa a restrições de energia. |
| "Células inteligentes..." | Linguagem antropomórfica e valorativa. | "Unidades de processamento cognitivo local..." | Termo puramente funcional e técnico. |
| "Morte celular consciente..." | Sugere vontade ou autoconsciência na abscisão. | "Desativação celular regulada por limiar energético (abscisão)..." | Alinha o termo com o equivalente biológico (apoptose/abscisão programada). |

---

## 3. Seção de Limitações Científicas e Ameaças à Validade

> [!WARNING]
> **Onde inserir:** Imediatamente antes da Conclusão, sob uma subseção dedicada de "Limitações e Trabalhos Futuros". Esta transparência metodológica desarma revisores rigorosos.

### Texto Sugerido (Português)
Embora os resultados demonstrem de forma estatisticamente rigorosa as vantagens da arquitetura *Digital Phytomer*, reconhecemos as seguintes limitações metodológicas e conceituais do presente estudo:

1. **Escala da População ($N=8$)**: A simulação avalia um enxame reduzido de 8 agentes cognitivos devido a restrições computacionais. Embora adequado para demonstrar dinâmicas de microssocialidade e plasticidade funcional local, as dinâmicas de emergência de grande escala (centenas ou milhares de agentes) podem diferir de forma não linear das curvas observadas.
2. **Utilização de LLMs Reduzidos (0.5B a 1.5B)**: O estudo foi parametrizado utilizando modelos locais de linguagem (Qwen 2.5:0.5b e Qwen 2.5:1.5b) para tornar viável a execução de 30 seeds independentes. Embora esses modelos demonstrem capacidades de inferência sob prompts direcionados, a consistência de parser e a robustez lógica de LLMs de escala de produção (ex. GPT-4, Gemini Pro) poderiam estabilizar ou mascarar as variações e falhas que acionam os mecanismos de trust e abscisão.
3. **Caráter Sintético das Transições de Domínio**: As quebras de paradigma (transições abruptas entre sequências lógicas, defesa cibernética, robótica e GSM8K) são representações formais discretas de estresse cognitivo. Em ambientes reais de produção, as transições ocorrem de forma fluida e mal-definida, o que exigiria um classificador semântico contínuo no barramento de eventos.
4. **Ausência de Grounding Físico**: Ao contrário dos fitômeros vegetais reais que interagem com pressões ecológicas e físicas em tempo real, as pressões ambientais do simulador são mediadas por taxas sintéticas de erro e penalidades monetárias de API. A transposição dessa analogia para o mundo físico (ex. enxames de drones reais) envolve ruídos de hardware e latências de rede não modelados nesta simulação.
5. **Restrição Epistemológica da Analogia Fisiológica**: A arquitetura do *Digital Phytomer* inspira-se em princípios de habituação, alostase e abscisão foliar. No entanto, alertamos contra qualquer interpretação que sugira a replicação da senciência ou cognição vegetal orgânica real. O modelo é puramente matemático-computacional, estruturado para extrair utilidade de analogias biológicas de controle distribuído.

---

## 4. Métodos Suplementares (Hiperparâmetros de Simulação)

> [!TIP]
> **Onde inserir:** Como Apêndice (Appendix A) do manuscrito ou na seção de Métodos Suplementares.

Para garantir a reprodutibilidade exata dos experimentos, consolidamos abaixo a ficha técnica de simulação:

* **Hardware de Referência**: Placa gráfica local GPU RTX 3050 Laptop (4GB VRAM), operando com servidor Ollama v0.1.48.
* **Modelo Padrão**: `qwen2.5:0.5b` (temperatura = 0.7, seed isolada por semente experimental).
* **Parâmetro de Confiança Inicial**: $\text{Trust}_{t=0} = 0.5$ (intervalo dinâmico $\in [0.0, 1.0]$).
* **Taxa Anti-Monopólio (Monopoly Tax)**: Penalidade exponencial de base $1.05$ aplicada à pontuação do leilão (bidding score) do agente para cada domínio individual após a resolução bem-sucedida de tarefas consecutivas naquele domínio.
* **Limiar de Abscisão (Depletion Threshold)**: Recursos iniciam em $100$ unidades. Sucessos somam $+10$ unidades de energia. Falhas locais subtraem $-5$ unidades de energia para o resolvedor principal e $-2$ unidades para os delegados associados. A abscisão ocorre quando o recurso chega a $\le 0$.
* **Sementes Experimentais**: 30 seeds independentes geradas no intervalo $[42, 71]$.
