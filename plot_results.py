import json
import matplotlib.pyplot as plt

with open('results_lr2.json', 'r') as f:
    all_costs = json.load(f)

plt.figure(figsize=(12, 8))

for hidden_layer_size, costs in all_costs.items():
    iterations, costs = zip(*costs)
    plt.plot(iterations, costs, label=f'Hidden nodes: {hidden_layer_size}')

plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Curves for Different Sizes of Hidden Layers')
plt.legend()
plt.grid(True)
plt.show()
