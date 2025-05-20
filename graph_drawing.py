import networkx as nx
import matplotlib.pyplot as plt

# Create a directed multigraph (allows self-loops and multi-edges)
G = nx.MultiDiGraph()

# Add nodes
G.add_nodes_from([1, 2, 3])

# Add edges (including self-loops and multi-edges)
G.add_edges_from([(1, 2), (1, 2), (2, 3), (3, 1), (1, 1)])

# Set up the plot
plt.figure(figsize=(8, 6))

# Draw the graph with curved edges
pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout

# Draw nodes and labels
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_labels(G, pos)

# Draw edges with curves (connectionstyle controls the curve)
curved_edges = [edge for edge in G.edges()]
nx.draw_networkx_edges(G, pos, edgelist=curved_edges, 
                      connectionstyle='arc3,rad=0.2',  # Curve the edges
                      arrowsize=15, width=1.5)

# Add self-loops with custom style
for node in G.nodes():
    if G.has_edge(node, node):
        # Different arc style for self-loops
        nx.draw_networkx_edges(G, pos, edgelist=[(node, node)], 
                              connectionstyle='arc3,rad=0.5',
                              arrowsize=15, width=1.5)
        nx.draw_networkx_edges(G, pos, edgelist=[(node, node)], 
                              connectionstyle='arc3,rad=0.4',
                              arrowsize=15, width=1.5)

plt.axis('off')  # Turn off the axis
plt.tight_layout()
plt.title("Graph with Curved Edges, Self-loops, and Multi-edges")
plt.show()