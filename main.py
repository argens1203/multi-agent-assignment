from simulate import Sim
from schelling import SchellingWorld

world = SchellingWorld(0.8, 0.7, 50, 50)
metrics = Sim(world).run()

# To actually see the plot, "Run Current File in Interative Window"
world.plot_metrics()
