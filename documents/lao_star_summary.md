# Summary Document: Core Principles of the LAO* Algorithm

### 1. The Three-Phase Loop: Expand → Test for Convergence → Cost Revision

The LAO* algorithm is an iterative three phase cycle to incrementally build and evaluate the solution until it stabilizes. It starts at **expand phase**. LAO* algorithm identifies a non-terminal "tip node" to represent a specific Python concept where the student has gaps. This node is in the current best partial solution, and we wants to expand it. This introduces its immediate successor states (more detailed foundation questions) into the searched graph. Then, the algorithm comes to **test for convergence phase**: after expansion, the algorithm checks whether the current best solution has reached a state of completion. It verifies if all reachable states under the current policy have been fully expanded and whether the error bounds have fallen below a predefined threshold ($\epsilon$). If yes, the search terminates. Finally, the **Cost Revision** happens. If the solution it found has not converged, the algorithm performs a localized ledger update using dynamic programming (Value Iteration or Policy Iteration). Crucially, instead of updating the entire state space, it only revises the costs and policies for a subset of states ($Z$) that includes the newly expanded node and its ancestors, which will update the estimated remaining cost (time) to complete the learning path.

### 2. What a "solution graph" is and how it differs from a search tree

**Search Tree** is A tree structure represents a strictly feed-forward progression where paths never merge and **no loops are allowed**. If a student fails a concept and needs to revisit it, a search tree cannot mook this "going back to review" behavior. This will cause standard tree-based algorithms to loop indefinitely or fail.

**Solution Graph** is a graph that **allows for loops**. In a Python learning project, a student's journey is rarely linear; it naturally involves cycles (Study -- Attempt -- Fail -- Re-study). A solution graph maps these cyclic dependencies into a network, enabling the system to keep a student in a learning loop until mastery is achieved.

### 3. How LAO* handles stochastic transitions (AND-nodes vs OR-nodes)

Since a student's answer to a quiz question is non-deterministic (ie., they may get it right or wrong with certain probabilities), the algorithm needs two distinct types of nodes to manage this uncertainty.

1. **OR-nodes (Choice Points)**: These noods represent the decisions made by the system. For instance, the system need to choose whether to give the student an "If-Else statement" question or a "For-Loop" question. The algorithm will choose the branch that **minimizes the overall expected cost**.

2. **AND-nodes (Probabilistic Outcomes)**: These noods represent the uncertainty (the student's random performance). For example, once a question is given, the student might have a 70% chance of passing and a 30% chance of failing. Since it is an AND-node, the algorithm must **account for all possible outcomes** simultaneously when calculating costs in order to ensure the system is prepared for both success and failure branches.

### 4. "Admissible heuristic" in this context and why it matters

In this context, a "heuristic" ($h(n)$) is the system's initial estimate of how many more mistakes a student will make before fully mastering the material. An "admissible" heuristic means this estimate **must always be optimistic**. It must never overestimate the true remaining cost to reach the goal.

If the system is too pessimistic about a student's potential, it might completely ignore a highly efficient learning path. By staying optimistic, the algorithm is mathematically incentivized to explore promising shortcuts. As the student actually answers questions, this optimism is gradually corrected by reality. It is the mathematical mechanism that allows LAO* to find the perfect path **without having to test the student on every single question in the database**.

### 5. Convergence Guarantee (Theorem 1) 

Theorem 1 provides the mathematical proof that the system **will** successfully find the absolute best, most time-saving assessment path. According to the theorem 1, LAO* is guaranteed to find an optimal solution under two conditions:

1. **Reachability**: An optimal policy must exist, meaning that it **must be physically possible** for the student to eventually reach the "mastery/goal state" from the initial starting state. There **cannot be any inescapable dead-ends.**

2. **Admissibility**: The initial heuristic values used to estimate the knowledge gaps for all un-unexplored states must be **optimistic**.



---



# Mapping Project Ariadne's SSP onto the LAO* Framework

In this project, the curriculum sequencing is modeled as a **Robust Hierarchical Stochastic Shortest Path (RH-SSP)** problem. This maps well with the LAO* algorithm.

To begin with, **States ($S$)** are nodes representing **conditions or configurations** in the explicit search graph. In Ariadne mapping, it is a **student's current knowledge state** or proficiency profile within the Knowledge Graph (KG). It encapsulates which Python concepts have been mastered, which are still unlearned, and potentially the historical tracking metrics. For example, they are latent traits being predicted from a deep knowledge tracing model.

What's more, **Actions ($A$)**  in LAO\* Definition represents the operations available to transition from one state to successor states, stemming from an **OR-node**. It is also corresponds to the pedagogical decision made by the system in Ariadne mapping. Specifically, it is an action to assign a particular learning curriculum component or quiz question to the student (eg., presenting a question about "for-loops"). The algorithm will search for the optimal sequence of these actions to minimize the expected path length to total mastery.

For **Transitions and Probabilities ($P$)**, it means the non-deterministic outcomes resulting from an action, represented as branches from an **AND-node** in LAO* definition. Moreover, it encapsulate the **stochastic nature of student performance** in Ariadne Mapping. When an action (eg. a quiz) is selected, the outcome is probabilistic(eg. a student has 80% chance to pass and transit to a "higher proficiency" state or otherwise a 20% chance they fail and loop back to a "review" state). These transitional probabilities are predicted by Ariadne's background models (like DKT/SAINT+).

In a massive Knowledge Graph, searching every possible learning sequence leads to a **state-space explosion**. Ariadne solves this by utilizing **ZPD (Zone of Proximal Development) Masking** to dramatically restrict LAO's forward expansion phase. The first step is to **prune irrelevant actions**. Instead of evaluating all possible educational content at an **OR-node**, the system applies a pedagogical mask derived from Vygotsky’s ZPD theory. It filters out actions that are either too easy (already mastered prerequisites) or too advanced (concepts whose prerequisites have not yet been unlocked). Moreover, **Constraining Graph Expansion** is also needed. During LAO's Expand phase, the algorithm only generates successor states that lie directly within the student's immediate learning frontier. As a result, this masking turns a mathematically unmanageable global MDP into a **localized, highly structured search**. LAO* only needs to find an optimal solution path within a narrow, pedagogically sound window, massively accelerating convergence while ensuring that the generated curriculum remains explainable and safe from cold-start anomalies.