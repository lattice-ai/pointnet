import plotly.graph_objects as go


def get_visualization_figure(mesh, label):
    x, y, z = mesh.numpy().T
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers'
            )
        ]
    )
    fig.update_layout(title=label)
    return fig
