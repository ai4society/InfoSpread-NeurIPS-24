# Dataset Description

This folder contains two versions of datasets, **Dataset v1** and **Dataset v2**, designed to examine different factors influencing the spread of misinformation in networked environments.

## Dataset v1
**Dataset v1** explores the impact of **network size** and the **initial count of infected nodes** on the spread of misinformation. The configurations include:
- **Network Sizes**: 10, 25, and 50 nodes
- **Initial Infected Nodes**: 1, 2, and 3 nodes for each network size

This setup results in a total of **9 unique datasets**. Each configuration contains **1000 random network states**, where opinion values of non-infected nodes are uniformly distributed between -0.5 and 0.6.

## Dataset v2
**Dataset v2** investigates the influence of the **initial connections (degrees of connectivity)** of infected nodes on misinformation spread. Similar to **Dataset v1**, networks consist of 10, 25, and 50 nodes. However, **Dataset v2** varies in the number of initial connections for infected nodes:
- **Degree of Connectivity**: Ranges from 1 to 4, where this degree refers to the number of `candidate nodes` connected to infected nodes at the start of the simulation.
  
This configuration produces a total of **12 unique datasets** for each setup, with **1000 network states per dataset**. The initial number of infected nodes is randomly chosen between 1 and 3. For example, a scenario with three initially infected nodes may still have a degree of connectivity of 1 if all infected nodes are linked to the same uninfected node.

## Directory Structure
- **Dataset_v1/**: Contains datasets with varying numbers of initially infected nodes.
- **Dataset_v2/**: Contains datasets with varying degrees of connectivity for initially infected nodes.
