# Binary Classification with Quantum Variational Circuit

The focus of this work is the construction of a quantum variational circuit capable of classifying classical data. That said, our system will consist of both a quantum and a classical component.  

- **Quantum Component (Quantum Circuit):** A block responsible for encoding/embedding, a block responsible for the variational algorithm, and measurement blocks.  

- **Classical Component:** Post-processing (associating measurements with labels), loss function computation, and optimization of variational parameters (θ) (see Fig. 1). In some cases, special classical preprocessing may also be necessary, such as dimensionality reduction of the dataset using PCA techniques [1].

These first two components (quantum circuit and post-processing) are connected in a loop, creating a hybrid system between quantum and classical computing.
