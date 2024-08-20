import plotly
from plotly.offline import iplot
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)


def plot_schelling(metrics):
    unhappiness, avg_similarity = metrics
    n = len(unhappiness)
    trace1 = go.Scatter(
        x=list(range(n)),
        y=unhappiness,
        mode="lines+markers",
        name="average unhappiness",
    )
    trace2 = go.Scatter(
        x=list(range(n)),
        y=avg_similarity,
        mode="lines+markers",
        name="average similarity",
    )
    data = [trace1, trace2]
    return iplot(data)
