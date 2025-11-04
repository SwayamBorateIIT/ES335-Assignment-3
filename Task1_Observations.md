# Comparative Analysis: Natural Language vs Structured Code Models

## Application

Here is the link to the Streamlit Application:

https://es335-assignment-3-u88kiqkunvq9h5jx2kaje2.streamlit.app/

## Summary

This analysis compares two NextWordMLP models trained on:
- *Dataset A (Category I)*: The Adventures of Sherlock Holmes - natural English text
- *Dataset B (Category II)*: Linux Kernel Code - structured C/C++ programming language

## 1. Dataset Characteristics

### Dataset A: Sherlock Holmes (Natural Language)

| Metric | Value |
|--------|-------|
| Total Tokens | 115,383 |
| Vocabulary Size | 8,175 (including PAD/UNK) |
| Most Common Token | '.' (6,333 occurrences) |
| Token Distribution | Natural language distribution with heavy emphasis on common words |

*Top 10 Tokens*: ., the, and, i, to, of, a, in, that, it

*Characteristics*:
- High redundancy with frequent function words
- Smaller vocabulary relative to token count
- Natural linguistic patterns and grammar
- Narrative structure with descriptive paragraphs

### Dataset B: Linux Kernel Code (Structured Language)

| Metric | Value |
|--------|-------|
| Total Tokens | 1,120,789 |
| Vocabulary Size | 43,161 (including PAD/UNK) |
| Most Common Token | '\n' (201,909 occurrences) |
| Token Distribution | Highly diverse with many unique identifiers |

*Top 10 Tokens*: \n, (, ), ;, ,, ->, =, *, {, }

*Characteristics*:
- **9.7** times more tokens than Dataset A
- **5.3** times larger vocabulary than Dataset A
- Dominated by structural/syntactic tokens
- Heavy use of identifiers, function names, and string literals
- Highly repetitive syntactic patterns

### Context Predictability

*Natural Language (Sherlock Holmes)*:
- Context depends heavily on semantic understanding
- Word choice influenced by narrative flow and style
- Higher ambiguity in next-word prediction
- Requires understanding of plot, character, and context

*Structured Code (Linux Kernel)*:
- Strong syntactic constraints (matching parentheses, semicolons)
- Predictable patterns (function declarations, loops, conditionals)
- Lower semantic ambiguity in structure
- Context windows can capture common code idioms

## 2. Model Performance Comparison

### Training Configuration
Both models used identical architectures:
- 2-layer MLP with 1024 hidden units each
- Block sizes: 3, 5
- Embedding dimensions: 32, 64
- Activations: ReLU, Tanh
- Primary comparison: emb_dim=64, activation=relu, block_size=5

### Loss Curves Analysis

<img width="1063" height="785" alt="image" src="https://github.com/user-attachments/assets/9a8f0d19-6e89-4cf0-8f0b-19059b97586c" />

<img width="1046" height="785" alt="image" src="https://github.com/user-attachments/assets/45e9697d-ce89-41d0-abb3-9a614466911c" />


#### Dataset A (Sherlock Holmes) - 500 epochs

Epoch   | Train Loss | Val Loss
--------|------------|----------
1       | 9.0103     | 8.6270
10      | 6.3977     | 6.4452
50      | 5.3639     | 5.9412
100     | 3.2777     | 6.9861
200     | 1.3942     | 9.6083
500     | 0.0537     | 16.4112


*Training Accuracy*: 90.21%

#### Dataset B (Linux Kernel) - 500 epochs

Epoch   | Train Loss | Val Loss
--------|------------|----------
1       | 6.7374     | 4.8358
10      | 2.9591     | 3.3273
50      | 1.8930     | 3.2635
100     | 1.5757     | 3.3903
200     | 1.3740     | 3.5816
500     | 1.2926     | 3.7558


*Training Accuracy*: 68.79%

### Key Observations

#### 1. Convergence Behavior

*Sherlock Holmes*:
- Lower initial loss (9.01 vs 6.74)
- Faster early convergence
- *Severe overfitting* from epoch ~60 onwards
- Final validation loss: 16.41 (severe overfitting)
- Gap between train/val: 16.36

*Linux Kernel*:
- Lower starting loss (better initial predictions)
- Steady, gradual improvement
- *Minimal overfitting* throughout training
- Final validation loss: 3.76 (stable)
- Gap between train/val: 2.46

#### 2. Reasons for Different Behaviours

*Natural Language Overfitting*:
- Model memorizes specific phrases and sentence structures
- 115K tokens insufficient for general language modeling
- High variance in natural language expressions
- Small dataset encourages memorization over generalization

*Code Generalization*:
- 1.12M tokens provide much more training data
- Repetitive syntactic patterns aid generalization
- Strong structural constraints reduce variance
- Many similar code patterns across codebase

#### 3. Training Accuracy vs Generalization

| Model | Train Accuracy | Overfitting Indicator |
|-------|---------------|---------------------|
| Sherlock Holmes | 90.21% | *Severe* (val loss 16.41) |
| Linux Kernel | 68.79% | *Minimal* (val loss 3.76) |

The lower training accuracy for Linux Kernel actually indicates *better generalization* - the model hasn't memorized the training set.

### Qualitative Generation Analysis

#### Sherlock Holmes (Temperature 0.8)

"Sherlock Holmes works to the other of the investigation which he closed 
keeping but his quietly. i shall just say time to you a turn which i have 
nothing in her due manner circle at the man said he. yes"


*Observations*:
- Grammatically weak structure
- Word being organised bit poorly
- Some recognizable phrases ("said he")
- Clear sign of overfitting - learned surface patterns not the actual meanings

#### Linux Kernel (Temperature 0.8)

"static inline int hrtick_enabled ( struct rq * rq , struct task_struct 
* curr = current ; if ( css_enable & ( 1 << 63 ) ) continue ; 
err = - EINVAL ; retval ="


*Observations*:
- *Syntactically valid* C code structure
- Proper function signature format
- Correct use of operators and keywords
- Reasonable variable naming conventions
- Code compiles structurally

## 3. Embedding Visualizations (t-SNE Analysis)

### Output Plots

<img width="1650" height="1362" alt="image" src="https://github.com/user-attachments/assets/1827b2d8-9b2b-4d82-9847-c026d7454044" />
<img width="1668" height="1362" alt="image" src="https://github.com/user-attachments/assets/8517d338-2011-4ac8-9426-eb642ae8f511" />

#### Natural Language
- Semantic clusters of verbs, nouns, adjectives grouped together
- Context-dependent positioning is present
- Grammatical relationships are maintained
- Character names potentially clustered

#### Structured Code
- Token type clusters of operators, keywords, identifiers
- Syntactic groupings of parentheses, brackets, semicolons
- Keyword clustes of if, for, while, return, static, void
- Type declarations grouped like for int, char, struct

### Visualization Insights

The t-SNE visualization for Linux Kernel shows selected tokens (int, return, if, for, static, void, =, ;) which likely form distinct clusters based on their syntactic roles:

- *Control flow keywords*: if, for, return
- *Type keywords*: int, void, static
- *Operators/Punctuation*: =, ;

Natural language embeddings would show more semantic relationships and less rigid clustering.

## 4. Summary: Natural vs Structured Language Learnability

### Why Structured Code Is More Learnable

| Factor | Natural Language | Structured Code | Better Stands |
|--------|------------------|-----------------|--------|
| *Syntactic Constraints* | Flexible grammar | Strict syntax rules | Code |
| *Repetitive Patterns* | High variability | Highly repetitive | Code |
| *Context Predictability* | Semantic-dependent | Syntax-dependent | Code |
| *Vocabulary Complexity* | Moderate, contextual | Large but patterned | Code |
| *Dataset Size* | 115K tokens | 1.12M tokens | Code |
| *Generalization* | Poor (overfits) | Good (generalizes) | Code |


### Practical Implications

*For Natural Language*:
- Requires much larger datasets (millions-billions of tokens)
- Benefits from pre-training on diverse texts
- Needs sophisticated architectures (Transformers with attention)
- Regularization critical to prevent overfitting

*For Structured Code*:
- More suited to smaller-scale models
- Syntax patterns learnable with simpler architectures
- Good results achievable with moderate dataset sizes
- Useful for code completion and syntax checking

## Conclusion

Structured programming languages are significantly more learnable than natural language for next-token prediction tasks due to:

1. *Deterministic syntax rules* that reduce ambiguity
2. *Repetitive patterns* that reinforce learning
3. *Larger effective training data* (10× more tokens)
4. *Lower contextual complexity* (syntax over meaning)

The natural language model's severe overfitting (train loss 0.05, val loss 16.41) versus the code model's stable generalization (train loss 1.29, val loss 3.76) clearly demonstrates that *programming languages provide better learning signals for neural models* at modest scale.

However, this doesn't mean natural language is harder - it's simply more *information-dense* and *dependence on context and meanings*, requiring orders of magnitude more data and more sophisticated architectures to model effectively.
